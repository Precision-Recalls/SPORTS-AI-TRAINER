import json
import logging

from azure.servicebus import ServiceBusMessage
from azure.servicebus.exceptions import ServiceBusError

logger = logging.Logger('CRITICAL')


def send_message_to_bus(sender, response, videoType, video_blob_name):
    try:
        json_response = json.dumps(response)
        message = ServiceBusMessage(json_response)
        message.application_properties = {"VideoType": videoType, "VideoID": video_blob_name}
        sender.send_messages(message)
    except ServiceBusError as e:
        logger.error(f"Failed to send message: {e}")
    except AttributeError as e:
        logger.error(f"Attribute error: {e}, possibly related to sender object initialization.")
