import time
import os
import sys
import logging
import anamoly

MAIN_LOOP_SLEEP_TIME = 0.5

def main():

    """ Initialize anamoly """
    
    # configures basic logger
    logger = logging.getLogger( __name__ )
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    handler.setFormatter(formatter)    
    logger.addHandler(handler)


    logger.info('Starting anamoly service ...')

    analytics = anamoly.DataAnalyzer(logger.name)
    analytics.handle_data()
    
    while True:
      time.sleep(MAIN_LOOP_SLEEP_TIME)


if __name__ == '__main__':
    main()
#