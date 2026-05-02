import os

QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/006539580860/async-ai-system-queue"
MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", "3"))
VISIBILITY_TIMEOUT = int(os.getenv("VISIBILITY_TIMEOUT", "30"))
WAIT_TIME = int(os.getenv("WAIT_TIME", "20"))
