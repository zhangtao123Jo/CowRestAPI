#!/usr/bin/env python
import os
from flask import Flask, abort, request, jsonify, g, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_httpauth import HTTPBasicAuth
from passlib.apps import custom_app_context as pwd_context
from itsdangerous import (TimedJSONWebSignatureSerializer
                          as Serializer, BadSignature, SignatureExpired)
import cv2
import numpy
import base64
from PIL import Image
from io import BytesIO
import utils
import json

# DOCS https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(4)

# initialization
app = Flask(__name__)
app.config['SECRET_KEY'] = 'the beijing telecom research center'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
app.config.base_images_path = 'f:/test_flask'

# extensions
db = SQLAlchemy(app)
auth = HTTPBasicAuth()


class User(db.Model):
    """
    The User class, userid/passwd and company_id are required.
    """
    __tablename__ = 'users'
    userid = db.Column(db.Integer, primary_key=True)
    company_id = db.Column(db.String(32), index=True)
    password_hash = db.Column(db.String(64))

    def hash_password(self, password):
        """
        Secure the passwd using hash encrypt.
        :param password:
        :return: the hash pwd
        """
        self.password_hash = pwd_context.encrypt(password)

    def verify_password(self, password):
        return pwd_context.verify(password, self.password_hash)

    def generate_auth_token(self, expiration=600):
        s = Serializer(app.config['SECRET_KEY'], expires_in=expiration)
        return s.dumps({'userid': self.userid})

    @staticmethod
    def verify_auth_token(token):
        s = Serializer(app.config['SECRET_KEY'])
        try:
            data = s.loads(token)
        except SignatureExpired:
            return None  # valid token, but expired
        except BadSignature:
            return None  # invalid token
        user = User.query.get(data['userid'])
        return user


class LogInfo(db.Model):
    __tablename__ = 'log_info'
    log_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    company_id = db.Column(db.String(32))
    remote_ip = db.Column(db.String(32))
    rfid_code = db.Column(db.String(32))
    imei = db.Column(db.String(64))
    extra_info = db.Column(db.String(64), nullable=True)

    def insert_record(self):
        try:
            db.session.add(self)
            db.session.commit()
        except:
            db.session.rollback()
            return False
        return True


@auth.verify_password
def verify_password(userid_or_token, password):
    # first try to authenticate by token
    user = User.verify_auth_token(userid_or_token)
    if not user:
        # try to authenticate with username/password
        user = User.query.filter_by(userid=userid_or_token).first()
        if not user or not user.verify_password(password):
            return False
    g.user = user
    return True


################################################
# Error Handling
################################################


@app.errorhandler(404)
def error_404(error):
    return str(error)


@app.errorhandler(405)
def error_405(error):
    return str(error)


@app.errorhandler(500)
def error_500(error):
    return str(error)


################################################
# Route Handling
################################################


@app.route('/api/add_user', methods=['POST'])
def new_user():
    """
    Add the user to the db.
    :return:
    """
    userid = request.json.get('userid')
    password = request.json.get('password')
    company_id = request.json.get('company_id')
    if userid is None or password is None or company_id is None:
        abort(400)  # missing arguments
    if User.query.filter_by(userid=userid).first() is not None:
        abort(400)  # existing user
    user = User(userid=userid, company_id=company_id)
    user.hash_password(password)
    db.session.add(user)
    db.session.commit()
    return (jsonify({'userid': user.userid, 'company_id': user.company_id}), 201,
            {'Location': url_for('get_user', userid=user.userid, _external=True)})


@app.route('/api/users/<int:userid>')
def get_user(id):
    """
    Get the id user.
    :param id:
    :return:
    """
    user = User.query.get(id)
    if not user:
        abort(400)
    return jsonify({'username': user.username})


@app.route('/api/token')
@auth.login_required
def get_auth_token():
    token = g.user.generate_auth_token(600)
    return jsonify({'token': token.decode('ascii'), 'duration': 600})


@app.route('/api/prospect', methods=['POST'])
@auth.login_required
def prospect():
    user_id = g.user.userid
    company_id = request.json.get('companyid')
    gather_time = request.json.get('gathertime')
    rfid_code = request.json.get('rfidcode')
    ip = request.json.get('ip')
    imei = request.json.get('imei')
    image_array = request.json.get('items')
    predict_array = []
    cid_array = []
    for i, item in enumerate(image_array):
        # get the base64 str and decode them to image
        img_base64 = item.get('cvalue')
        img_oriented = item.get('cid')
        starter = img_base64.find(',')
        img_base64 = img_base64[starter + 1:]
        bytes_buffer = BytesIO(base64.b64decode(img_base64))
        # get one for pillow and another for
        img_pillow = Image.open(bytes_buffer)
        img_cv2 = cv2.imdecode(numpy.frombuffer(bytes_buffer.getvalue(), numpy.uint8), cv2.IMREAD_COLOR)
        predict_array.append(img_pillow)
        cid_array.append(img_oriented)

    # get the previous registered images top3 and encode them to base64
    pre_files = utils.get_files(os.path.join(app.config.base_images_path, company_id, rfid_code) + os.sep, 3)
    pre_items = utils.read_image_to_base64(pre_files)
    # get the predicted results and returned
    result = utils.get_predicted_result(predict_array, cid_array)
    return jsonify({
        'userid': user_id,
        'companyid': company_id,
        'resoult': 1,
        'gather_time': gather_time,
        'percent': result,
        'verinfo': 'test',
        'ip': ip,
        'imei': imei,
        'items': pre_items
    })


@app.route('/api/verify', methods=['POST'])
@auth.login_required
def verify():
    # get the params first
    user_id = g.user.userid
    json_obj = json.loads(request.form.get('entity'))
    company_id = json_obj.get('companyid')
    gather_time = json_obj.get('gathertime')
    rfid_code = json_obj.get('rfidcode')
    ip = json_obj.get('ip')
    imei = json_obj.get('imei')
    video = request.files['video']
    # make async execution thread for the video save and frame grabber
    executor.submit(utils.process_video_to_image, video,
                    os.path.join(app.config.base_images_path, company_id, rfid_code) + os.sep, rfid_code)
    # log the submit info to the db
    li = LogInfo(company_id=company_id, rfid_code=rfid_code, remote_ip=ip, imei=imei,
                 extra_info='file name is : ' + video.filename)
    li.insert_record()

    return jsonify({
        'userid': user_id,
        'companyid': company_id,
        'resoult': True,
        'gather_time': gather_time,
        'verinfo': 'jobs was launched in background',
        'ip': ip,
        'imei': imei,
    })


#
if __name__ == '__main__':
    if not os.path.exists('db.sqlite'):
        db.create_all()
    app.run(host='0.0.0.0', debug=True)
