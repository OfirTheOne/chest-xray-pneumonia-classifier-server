
FROM tiangolo/uwsgi-nginx-flask:python3.6-alpine3.7

ENV PORT = 8080

RUN apk --update add bash nano
RUN pip install --upgrade

COPY ./ ./app
WORKDIR ./app

COPY ./requirements.txt /var/www/requirements.txt
RUN pip install -r /var/www/requirements.txt

CMD ["python", "app.py", "${PORT}"]