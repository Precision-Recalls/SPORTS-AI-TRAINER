import logging

from azure.servicebus import ServiceBusMessage
from azure.servicebus.exceptions import ServiceBusError

logger = logging.Logger('CRITICAL')


def send_message_to_bus(sender, response, videoType, video_blob_name):
    try:
        message = ServiceBusMessage(str(response))
        message.application_properties = {"MessageType": videoType, "VideoID": video_blob_name}
        sender.send_messages(message)
    except ServiceBusError as e:
        logger.error(f"Failed to send message: {e}")
