import requests
import json
import numpy as np
import cv2


# use this script to test out pinging the mc instance server

response = requests.post(
    'http://localhost:8000/take_step',
    json={
        'action': [0, 1, 0, 12, 12, 0, 0, 0],
    }
)

return_dict = eval(response.content.decode())
arr = np.asarray(eval(return_dict['obs'])).astype(np.uint8)
arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
cv2.imwrite('test_received.png', arr)