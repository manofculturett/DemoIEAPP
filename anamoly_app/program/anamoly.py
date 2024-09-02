import paho.mqtt.client as mqtt
import sys
import logging
# import statistics
import json
import numpy as np
from sklearn.ensemble import IsolationForest

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
        self.model = IsolationForest(contamination=0.03)

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
            self.logger.info('new message: {}'.format(new_msg))
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
        self.logger.info('Anomaly detection started')
        anomaly_state = False
        data = [entry['_value'] for entry in payload]
        data = np.array(data).reshape(-1,1)
        self.model.fit(data)
        y_pred = self.model.predict(data)
        if -1 in y_pred:
            self.logger.info('Anomaly detected')
            anomaly_state = True
        else:
            self.logger.info('No anomaly detected')
        self.client.publish('anomaly', payload=json.dumps({'anomaly':anomaly_state}))



    def handle_data(self):
        try:
            self.client.username_pw_set(USERNAME, PASSWORD)
            self.client.connect(BROKER_ADDRESS)
            self.client.loop_start()
            self.logger.info('Subscribe to handling topic')
            self.subscribe('handling',callback=self.anomaly)

        except Exception as err:
            self.logger.error('An error ocurred while handling data: {}'.format(err))