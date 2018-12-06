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
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
app.config['SQLALCHEMY_DATABASE_URI']='mysql+mysqlconnector://root:123456@localhost:3306/cowrest'
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
# app.config.base_images_path = 'f:/test_flask'
app.config.base_images_path = 'd:/cowrest_test'

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
    """
    The log class
    """
    __tablename__ = 'log_info'
    log_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    company_id = db.Column(db.String(32))
    remote_ip = db.Column(db.String(32))
    rfid_code = db.Column(db.String(32))
    imei = db.Column(db.String(64))
    extra_info = db.Column(db.String(64), nullable=True)


class Archives(db.Model):
    """
    The archives class
    """
    __tablename__ = 'archives'
    aid = db.Column(db.Integer, primary_key=True, autoincrement=True)
    rfid_code = db.Column(db.String(32))
    age = db.Column(db.Integer)
    company_id = db.Column(db.String(32))
    gather_time = db.Column(db.String(32))
    folder_path = db.Column(db.String(200))
    health_status = db.Column(db.String(32))
    extra_info = db.Column(db.String(64), nullable=True)


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



@app.errorhandler(400)
def error_400(error):
    return jsonify({"status":"1x0000","message":"{} param not right".format(error.description)})


@app.errorhandler(403)
def error_403(error):
    return jsonify({"status":"5x0001","message":"Data already exists"})


@app.errorhandler(404)
def error_404(error):
    return str(error)


@app.errorhandler(405)
def error_405(error):
    return str(error)


@app.errorhandler(500)
def error_500(error):
    return str(error)


@app.errorhandler(502)
def error_502(error):
    return jsonify({"status":"3x0000","message":"database operation error"})


###############################################
# Route Handling
###############################################


@app.route('/api/add_user', methods=['POST'])
def new_user():
    """
    Add the user to the db.
    :return:
    """
    userid = request.json.get('userid')
    password = request.json.get('password')
    company_id = request.json.get('company_id')
    # verify the existence of parameters
    utils.verify_param(abort,error_code=400,userid=userid,password=password,company_id=company_id)
    if User.query.filter_by(userid=userid).first() is not None:
        abort(403)  # existing user
    user = User(userid=userid, company_id=company_id)
    user.hash_password(password)
    db.session.add(user)
    db.session.commit()
    return (jsonify({'userid': user.userid, 'company_id': user.company_id}), 201,
            {'Location': url_for('get_user', userid=user.userid, _external=True)})


@app.route('/api/users/<int:userid>')
def get_user(userid):
    """
    Get the user by id.
    :param userid:
    :return:
    """
    user = User.query.get(userid)
    utils.verify_param(abort,error_code=502,user=user)
    return jsonify({'username': user.userid})


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
    # verify the existence of parameters
    utils.verify_param(abort,error_code=400,user_id=user_id,company_id=company_id,gather_time=gather_time,rfid_code=rfid_code,ip=ip,imei=imei,image_array=image_array)
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
        'gathertime': gather_time,
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
    try:
        video = request.files['video']
    except:
        video=None
    # give the age value, 0 for default now.
    age = 0  # json_obj.get("age")
    # give the health_status value, 1 for default now.
    health_status = '1'  # json_obj.get("health_status")
    #verify the existence of parameters
    utils.verify_param(abort,error_code=400,user_id=user_id,json_obj=json_obj,company_id=company_id,gather_time=gather_time,rfid_code=rfid_code,ip=ip,imei=imei,video=video,age=age,health_status=health_status)

    if Archives.query.filter_by(rfid_code=rfid_code).first():
        abort(403)
    else:
        # make the save folder path and save the video
        folder_path = os.path.join(app.config.base_images_path, company_id, rfid_code) + os.sep
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        video.save(folder_path + utils.secure_filename(video.filename))

        # make async execution thread for the video save and frame grabber
        executor.submit(utils.process_video_to_image, video,
                        os.path.join(app.config.base_images_path, company_id, rfid_code) + os.sep, rfid_code)
        # assign values to database fields
        archives = Archives(rfid_code=rfid_code, age=age, company_id=company_id, gather_time=gather_time,
                            health_status=health_status,
                            folder_path=os.path.join(app.config.base_images_path, company_id, rfid_code),
                            extra_info='file name is : ' + video.filename)
        li = LogInfo(company_id=company_id, rfid_code=rfid_code, remote_ip=ip, imei=imei,
                     extra_info='file name is : ' + video.filename)
        db_list = [archives, li]
        #log the submit info to the db
        utils.insert_record(db_list, db,abort)
        return jsonify({
            'userid': user_id,
            'companyid': company_id,
            'resoult': True,
            'gathertime': gather_time,
            'verinfo': 'jobs was launched in background',
            'ip': ip,
            'imei': imei,
        })


@app.route('/api/list', methods=['POST'])
@auth.login_required
def cow_list_by_company_id():
    """
     Query the cows of the corresponding company
    :return: cow_list
    """
    def json_serilize(instance):
        """
         change the returned data to dict
        :param instance: data
        :return:
        """
        return {
            "userid":g.user.userid,
            "aid": instance.aid,
            "rfid_code": instance.rfid_code,
            "age": instance.age,
            "company_id": instance.company_id,
            "gather_time": instance.gather_time,
            "health_status": instance.health_status,
            "extra_info": instance.extra_info,
            "folder_path": instance.folder_path
        }
    current_page=request.json.get("currentpage")
    cow_number=request.json.get("cownumber")
    company_id = request.json.get("companyid")
    utils.verify_param(abort,error_code=400,company_id=company_id)
    try:
        #get a list of cowrest based on the current page number and display number
        if current_page and cow_number:
            cow_list = Archives.query.filter_by(company_id=company_id).paginate(page=current_page, per_page=cow_number).items
        else:
            # return all cows list without current page or display number
            cow_list=Archives.query.filter_by(company_id=company_id).all()
        return json.dumps(cow_list, default=json_serilize)
    except:
        abort(502)


@app.route('/api/verify_cow_exists', methods=['POST'])
@auth.login_required
def verify_cow_exists():
    """
     verify the existence of cows
    :return:
    """
    user_id = g.user.userid
    company_id = request.json.get('companyid')
    rfid_code = request.json.get('rfidcode')

    utils.verify_param(abort,error_code=400,user_id=user_id,company_id=company_id,rfid_code=rfid_code)
    if Archives.query.filter_by(rfid_code=rfid_code,company_id=company_id).first():
        result=True
    else:
        result=False
    return jsonify({
        'userid': user_id,
        'companyid': company_id,
        'rfid_code':rfid_code,
        'result':result
    })


# the main entry when using flask only, of course you should use uwsgi instead in deploy environment.
if __name__ == '__main__':
    if not os.path.exists('db.sqlite'):
        db.create_all()
    app.run(host='0.0.0.0', debug=True)