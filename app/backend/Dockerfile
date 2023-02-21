# Build the app image
FROM python:3.10
# Create directory for the app user
RUN mkdir -p /home/app

# Create the app user
RUN groupadd app && useradd -g app app

# Create the home directory
ENV APP_HOME=/home/app/backend
RUN mkdir -p $APP_HOME
WORKDIR $APP_HOME

# install
COPY . $APP_HOME
RUN pip install -r requirements.txt

RUN chown -R app:app $APP_HOME
USER app
CMD ["uvicorn","pamps.app:app","--host=0.0.0.0","--port=8000","--reload"]