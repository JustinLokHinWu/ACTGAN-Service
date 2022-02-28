# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

EXPOSE 5000

WORKDIR /app

COPY * /app/

RUN pip install -r requirements.txt

CMD if $DEBUG ; then \
        python app.py ; \
    else \
        ["python3", "-m" , "flask", "run", "--host=0.0.0.0"] ; \
    fi

# CMD python app.py
# CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
# CMD python -m flask run