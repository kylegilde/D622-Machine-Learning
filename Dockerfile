FROM python:3.6.5

WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN git clone https://github.com/kylegilde/D622-Machine-Learning /usr/src/app/D622-Machine-Learning
EXPOSE 5000
CMD [ "python", "/usr/src/app/D622-Machine-Learning/main.py" ]