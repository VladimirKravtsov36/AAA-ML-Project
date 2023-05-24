from locust import HttpUser, task, between


class MyUser(HttpUser):
    wait_time = between(1, 3)  # Wait time between simulated requests

    @task
    def replace_background(self):
        # Define the file path of the image to upload
        image_path = 'test_images/standard.jpg'

        # Define the payload with the file to be uploaded
        files = {'image': open(image_path, 'rb')}

        # Send the POST request to the endpoint
        self.client.post('/', files=files)

# locust -f load_test.py --host http://84.252.128.219:8000