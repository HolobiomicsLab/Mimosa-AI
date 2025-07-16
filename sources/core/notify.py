"""
Optional class for push notifications using Pushover API to a phone.
This class allows sending push notifications with a message, title, and priority.
Useful for getting a notification when a workflow is completed or fails.
"""

import http.client
import os
import urllib.parse


class PushNotifier:
    def __init__(self, token, user):
        self.token = token
        self.user = user
        self.host = "api.pushover.net"
        self.port = 443

    def send_message(self, message, title=None, priority=0):
        """Send a push notification message."""
        if not self.token or not self.user:
            return None
        data = {
            "token": self.token,
            "user": self.user,
            "message": message,
            "priority": priority,
        }
        if title:
            data["title"] = title
        conn = http.client.HTTPSConnection(f"{self.host}:{self.port}")
        conn.request(
            "POST",
            "/1/messages.json",
            urllib.parse.urlencode(data),
            {"Content-type": "application/x-www-form-urlencoded"},
        )

        response = conn.getresponse()
        conn.close()
        return response


if __name__ == "__main__":
    token = os.getenv("PUSHOVER_TOKEN")
    user = os.getenv("PUSHOVER_USER")
    if not token or not user:
        print("Pushover token or user not set in environment variables.")
    else:
        notifier = PushNotifier(token, user)
        response = notifier.send_message(
            "Hello from Mimosa!", title="Mimosa Notification", priority=1
        )
        print(f"Notification sent: {response.status} {response.reason}")
