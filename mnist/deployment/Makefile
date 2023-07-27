bento_service_run:
	bentoml serve service:mnist_service --reload

bento_containerize:
	bentoml containerize mnist_classifier:latest

docker_run:
	docker run -p 3000:3000 mnist_classifier

