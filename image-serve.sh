docker image build -t server-app .
docker container run -d -p 80:80 server-app