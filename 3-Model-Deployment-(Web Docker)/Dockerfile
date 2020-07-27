FROM python:3.7

#copy the corrent directory contents into the container at /web_app
ADD . /web_app

#set the directory to /web_app
WORKDIR /web_app

# copy the requirements into the container at /etc
COPY ./requirements.txt /etc

# install any needed packages specified in requirements
RUN pip install -r /etc/requirements.txt

#Make port 5000 available outside the container
EXPOSE 5000

ENTRYPOINT ["python"]
CMD ["web_app.py"]
