import paho.mqtt.client as mqtt
import sys
import logging
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

BROKER_ADDRESS = 'ie-databus'
BROKER_PORT = 1883
MICRO_SERVICE_NAME = 'data-analytics'
USERNAME = 'edge'
PASSWORD = 'edge'

class DeepLog(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(DeepLog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Only take the output of the last time step
        out = self.fc(out)
        return out

class DataAnalyzer():
    """
    Data Analyzer connects to MQTT broker and waits for new
    input data to calculate KPIs and perform anomaly detection.
    """

    def __init__(self, logger_parent):
        logger_name = '{}.{}'.format(logger_parent, __name__)
        self.logger = logging.getLogger(logger_name)
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_subscribe = self.on_subscribe
        self.client.on_message = self.on_message
        self.topic_callback = dict()
        self.is_training = False
        self.threshold = 6

        # Initialize DeepLog model
        self.input_size = 1  # num of feature
        self.hidden_size = 64
        self.num_layers = 2
        self.output_size = 1  
        self.model = DeepLog(self.input_size, self.hidden_size, self.num_layers, self.output_size)

        # Define loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # For normalizing the input data
        self.scaler = MinMaxScaler()

    def on_connect(self, client, userdata, flags, rc):
        self.logger.info('Connected successfully to broker, response code {}'.format(rc))

    def on_disconnect(self, client, userdata, rc):
        if rc != 0:
            self.logger.warning('Connection ended unexpectedly from broker, error code {}'.format(rc))

    def on_subscribe(self, client, userdata, mid, granted_qos):
        self.logger.info('Successfully subscribed')

    def on_message(self, client, userdata, message):
        self.logger.info('New message received on topic: {}'.format(message.topic))
        if message.topic != 'ie':
            new_msg = json.loads(message.payload)
            try:
                self.topic_callback[message.topic](new_msg)
            except Exception as err:
                self.logger.error('An error occurred while handling new message of {}: {}'.format(message.topic, err))

    def subscribe(self, topic, callback):
        """ Subscribes to given topic, assigning a callback function that
        will be called when a new message is received """
        self.topic_callback[topic] = callback
        self.client.subscribe(topic)

    def prepare_data(self, payload, sequence_length=5):
        """ Prepare data as sequences for LSTM input """
        data = [entry['_value'] for entry in payload]
        data = np.array(data).reshape(-1, 1)
        data = self.scaler.fit_transform(data)  # Normalize the data

        sequences = []
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:i + sequence_length])

        return np.array(sequences)

    def train_deeplog(self, payload, sequence_length=5, num_epochs=15):
        """ Train DeepLog model on normal data """
        if self.is_training:
            self.logger.warning('DeepLog model is already training')
            return
        try:
            self.is_training = True
            self.logger.info('Training DeepLog model...')
            sequences = self.prepare_data(payload, sequence_length)
            X_train = torch.tensor(sequences[:-1], dtype=torch.float32)
            y_train = torch.tensor(sequences[1:], dtype=torch.float32)
            for epoch in range(num_epochs):
                self.model.train()
                self.optimizer.zero_grad()
                outputs = self.model(X_train)
                loss = self.criterion(outputs, y_train[:, -1, :])  # Only predict the last time step
                loss.backward()
                self.optimizer.step()

                if (epoch + 1) % 2 == 0:
                    self.logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            self.logger.info('Model training completed')
            self.client.publish('anomaly', payload=json.dumps({'status': 'completed'}))
        except Exception as err:
            self.logger.error('An error occurred while training DeepLog model: {}'.format(err))
            self.client.publish('anomaly', payload=json.dumps({'status': 'error: {}'.format(err)}))
        finally:
            self.is_training = False

    def anomaly(self, payload):
        sequence_length = 5
        self.logger.info('Anomaly detection started using DeepLog')

        # Prepare the data for testing
        sequences = self.prepare_data(payload, sequence_length)
        X_test = torch.tensor(sequences, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test).numpy()

        # Rescale the predictions back to original scale
        predictions = self.scaler.inverse_transform(predictions)

        anomaly_entries = []
        threshold =  self.threshold # Set a threshold for anomaly detection
        for i, entry in enumerate(payload[sequence_length:]):
            actual_value = entry['_value']
            predicted_value = predictions[i, 0]

            if abs(actual_value - predicted_value) > threshold:
                anomaly_entries.append(entry)

        # Log and publish anomaly results
        if anomaly_entries:
            self.logger.info(f'Anomalies detected')
            self.client.publish('anomaly', payload=json.dumps({'anomaly': True, 'entries': anomaly_entries}))
        else:
            self.logger.info('No anomalies detected')
            self.client.publish('anomaly', payload=json.dumps({'anomaly': False}))
    def handle_training(self, payload):
        """ Handle training data from a different topic """
        self.logger.info('Received training data, starting model training')
        self.train_deeplog(payload)
    def set_threshold(self, payload):
        self.threshold = payload['threshold']
        self.logger.info(f'Threshold set to {self.threshold}')
        self.client.publish('anomaly', payload=json.dumps({'threshold': self.threshold}))


    def handle_data(self):
        try:
            self.client.username_pw_set(USERNAME, PASSWORD)
            self.client.connect(BROKER_ADDRESS)
            self.client.loop_start()
            self.subscribe('deeplogtrain', callback=self.handle_training)
            self.logger.info('Subscribed to deeplogtrain topic')
            self.subscribe('handling', callback=self.anomaly)
            self.logger.info('Subscribed to handling topic')
            self.subscribe('thresholdset', callback=self.set_threshold)

        except Exception as err:
            self.logger.error('An error occurred while handling data: {}'.format(err))
