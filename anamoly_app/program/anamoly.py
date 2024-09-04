import paho.mqtt.client as mqtt
import sys
import logging
# import statistics
import json
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

BROKER_ADDRESS='ie-databus'
BROKER_PORT=1883
MICRO_SERVICE_NAME = 'data-analytics'
""" Broker user and password for authtentification"""
USERNAME='edge'
PASSWORD='edge'

class DataAnalyzer():
    """
    Data Analyzer connects to mqtt broker and waits for new
    input data to calculate KPIs.

    """

    def __init__(self, logger_parent):
        """ Starts the instantiated object with a proper logger """
        
        logger_name = '{}.{}'.format(logger_parent,__name__)
        self.logger = logging.getLogger(logger_name)
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_subscribe = self.on_subscribe
        self.client.on_message = self.on_message
        self.topic_callback = dict()
        self.model_iso = IsolationForest(n_estimators=200,contamination=0.01,n_jobs=-1)
        self.model_lof = LocalOutlierFactor(n_neighbors=2, contamination=0.01,n_jobs=-1)

    def on_connect(self, client, userdata, flags, rc):
        self.logger.info('Connected successfully to broker, response code {}'.format(rc))

    def on_disconnect(self, client, userdata, rc):
        if rc != 0:
            self.logger.warning('Connection ended unexpectedly from broker, error code {}'.format(rc))


    def on_subscribe(self, client, userdata, mid, granted_qos):
        
        self.logger.info('successfully subscribed ')

    def on_message(self, client, userdata, message):
        self.logger.info('New message received on topic: {}'.format(message.topic))
        if message.topic != 'ie':
            new_msg = json.loads(message.payload)
            # self.logger.info('new message: {}'.format(new_msg))
            try:
                self.topic_callback[message.topic](new_msg)
            except Exception as err:
                self.logger.error('An error ocurred while hanlding new message of {}: {}'.format(message.topic, err))

    def subscribe(self, topic, callback):
        """ Subscribes to given topic, assigning a callback function that
        will be called when a new message is received """
        self.topic_callback[topic] = callback
        self.client.subscribe(topic)

    def anomaly(self, payload):
        anomaly_entries = []  # Store all anomaly entries
        self.logger.info('Anomaly detection started')
        anomaly_state = False

        # Extract data and reshape for the model
        data = [entry['_value'] for entry in payload]
        data = np.array(data).reshape(-1, 1)

        # Fit the models and get predictions
        self.model_iso.fit(data)
        self.model_lof.fit(data)
        anomalies_isotree = self.model_iso.predict(data)
        anomalies_lof = self.model_lof.fit_predict(data)

        self.logger.info('Anomaly results - Isolation Forest: {} | LOF: {}'.format(anomalies_isotree, anomalies_lof))

        # Check for anomalies in both models
        for i in range(len(data)):
            if anomalies_isotree[i] == -1 and anomalies_lof[i] == -1:
                anomaly_state = True
                anomaly_entries.append(payload[i])  # Collect all anomalies

        # Log anomaly detection result
        if anomaly_state:
            self.logger.info('Anomaly detected in both models')
            self.logger.info('Anomaly entries: {}'.format(anomaly_entries))
        else:
            self.logger.info('No anomalies detected')
        # self.logger.info('Anomaly entries: {}'.format(anomaly_entries))
        # Publish the anomaly detection result
        self.client.publish('anomaly', payload=json.dumps({'anomaly': anomaly_state, 'entries': anomaly_entries}))



    def handle_data(self):
        try:
            self.client.username_pw_set(USERNAME, PASSWORD)
            self.client.connect(BROKER_ADDRESS)
            self.client.loop_start()
            self.subscribe('handling',callback=self.anomaly)
            self.logger.info('Subscribe to handling topic')

        except Exception as err:
            self.logger.error('An error ocurred while handling data: {}'.format(err))