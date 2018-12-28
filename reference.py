from flask import Flask, request, render_template, redirect, url_for, Response
from werkzeug.utils import secure_filename
import create_xml
import faster.tools.demo as demo
import json
import cv2
import tensorflow as tf
import numpy
import logging
import logging.config
import time
import config
import util
import numpy as np
import threading 
from keras.applications.imagenet_utils import preprocess_input

global app 
app = Flask(__name__)

#application = app.wsgifunc()

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'bmp']

logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s] %(levelname)s [%(funcName)s: %(filename)s, %(lineno)d] %(message)s',
                    filename="logging-xinniuren.logs",
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filemode='a')


def allowed_file(filename):
    # filename:'a.dog.png'  ->  filename.rsplit('.') : ['a', 'dog', 'png']
    return '.' in filename and filename.rsplit('.')[-1].lower() in ALLOWED_EXTENSIONS


################################################
# Error Handling
################################################


@app.errorhandler(404)
def error_404(error):
    return render_template("error.html")


@app.errorhandler(405)
def error_405(error):
    return render_template("error.html")


@app.errorhandler(500)
def error_500(error):
    return str(error)


################################################
# about
################################################
@app.route("/about/")
def about():
    return render_template("about.html")


# route operation
@app.route('/', methods=['POST', "GET"])
def root():
    return render_template("index.html")


# @app.route('/json')
# def json():
#     return json.dumps({'xml':})

def get_image(file_location):
    """
    :param file_location: image path
    :return: img, img.shape
    """
    fname = file_location
    print(fname)
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    print('img_shape:', img.shape)
    if img is None:
        return None
    return img, img.shape


@app.route("/upload_image", methods=['GET', 'POST'])
def upload_image():
    
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(url_for("root"))

        file = request.files['file']
        ip = request.remote_addr
        ######################################
        # get the value of multi_label_open from front page
        # multi_label_open = request.files['file']

        # False : indicate return multi label
        # True : indicate return '1'  default is True
        ######################################
        # False : indicate return multi label
        # True : indicate return '1'  default is True
        signal = str(request.form['identified'])

        prediction_result = ''
        top3_dic_names = {}

        # if user does not select file, browser also submit a empty part without filename
        if file.filename == '':
            return redirect(url_for("root"))

        if file and allowed_file(file.filename):
            filestr = request.files['file'].read()
            logging.info('From ' + ip + ' -> Upload image ' + file.filename + ' with size of {} kb'.format(
                len(filestr) / float(1000.0)))
            try:
                # convert string data to numpy array
                npimg = numpy.fromstring(filestr, numpy.uint8)
                # convert numpy array to image
                img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                imagename = secure_filename(file.filename).lower()  # '7.jpeg'
                imageshape = img.shape  # (1440, 2560, 3)
            except:
                print('Corrupt Image:',file.filename)
                logging.info('Corrupt Image:' + file.filename)
                return 'error'
            # filename : is image path such as 'static/img_pool/7.jpeg'
            # filename = os.path.join("static/img_pool", secure_filename(file.filename).lower())
            start = time.clock()

            executor = APPEntity()

            if signal == '0':
                all_coordinates = executor.predict_obj(img)
                margin = 15
                for arry in all_coordinates:
                    arry[-4] -= margin
                    arry[-3] -= margin
                    arry[-2] += margin
                    arry[-1] += margin
                '''create xml'''
                prediction_result = create_xml.create_xml(imagename, imageshape, all_coordinates, {})

            elif signal == '1':
                all_coordinates = executor.predict_obj(img)
                if len(all_coordinates) != 0:
                    margin = 10
                    for arry in all_coordinates:
                        arry[-4] -= margin
                        arry[-3] -= margin
                        arry[-2] += margin
                        arry[-1] += margin

                    all_coordinates, top3_dic_names = executor.predict_face(img, all_coordinates)
                prediction_result = create_xml.create_xml(imagename, imageshape, all_coordinates, top3_dic_names)

            elif signal == '2':
                all_coordinates = json.loads(request.form.get('coordinates'))
                all_coordinates_trans = []
                if len(all_coordinates) != 0:
                    for arry in all_coordinates:
                        arry_trans = [arry['Id'], arry['X'], arry['Y'], arry['X'] + arry['W'], arry['Y'] + arry['H']]
                        all_coordinates_trans.append(arry_trans)
                        all_coordinates, top3_dic_names = executor.predreict_back(img, all_coordinates_trans)
                prediction_result = create_xml.create_xml(imagename, imageshape, all_coordinates, top3_dic_names, need_id = True)

            elif signal == '3':
                _, all_coordinates_face = demo.demo(sess, net, img)
                top3_dic_names_face = {}
                if len(all_coordinates_face) != 0:
                    margin = 10
                    for arry in all_coordinates_face:
                        arry[-4] -= margin
                        arry[-3] -= margin
                        arry[-2] += margin
                        arry[-1] += margin
                        all_coordinates_face, top3_dic_names_face = executor.predict_face(img, all_coordinates_face)
                prediction_result_face = create_xml.create_xml(imagename, imageshape, all_coordinates_face, top3_dic_names_face)

                all_coordinates = json.loads(request.form.get('coordinates'))
                all_coordinates_trans = []
                top3_dic_names_back = {}
                all_coordinates_back = []
                if len(all_coordinates) != 0:
                    for arry in all_coordinates:
                        arry_trans = [arry['Id'], arry['X'], arry['Y'], arry['X'] + arry['W'], arry['Y'] + arry['H']]
                        all_coordinates_trans.append(arry_trans)
                        all_coordinates_back, top3_dic_names_back = executor.predict_back(img, all_coordinates_trans)
                prediction_result_back = create_xml.create_xml(imagename, imageshape, all_coordinates_back, top3_dic_names_back, need_id = True)

                prediction_result = prediction_result_face+prediction_result_back

            elif signal == '4':
                recognized_class, score = executor.predict_antnet(img)
                prediction_result = recognized_class+","+str(score)

            end = time.clock()
            logging.info('From ' + ip + ' -> Deal with ' + file.filename + ' using GPU : {} s'.format(end - start))
            # return render_template("index.html", prediction_result=prediction_result)
            return prediction_result

    return redirect(url_for("root"))


class APPEntity(object):
    sess = None
    net = None
    graph_objdetect = None
    keras_model_face = None
    classes_cows_face = None
    graph_face = None
    keras_model_back = None
    classes_cows_back = None
    graph_back = None
    keras_model_antnet = None
    classes_cows_antnet = None
    graph_antnet = None

    def __init__(self):
        # im, all_coordinates = demo.detection_demo(im_names)
        global sess, net, keras_model_antnet,keras_model_back,keras_model_face,\
            classes_cows_back,classes_cows_antnet,classes_cows_face, graph_face, graph_back,graph_antnet,graph_objdetect
        sess, net = demo.restore_model('60000')
        graph_objdetect = tf.get_default_graph()
        # tf.reset_default_graph()
        # sess_id, net_id = demo_id.restore_model('100000')

        # set the backend whether tf or theano
        util.set_img_format()

        config.model = 'xception_face'
        # get the keras original model object
        keras_module_face = util.get_model_class_instance()
        # load the best trained model
        keras_model_face = keras_module_face.load()
        graph_face = tf.get_default_graph()
        # get the classes dict
        classes_cows_face = util.get_classes_in_keras_format()

        config.model = 'xception_back'
        # get the keras original model object
        keras_module_back = util.get_model_class_instance()
        # load the best trained model
        keras_model_back = keras_module_back.load()
        graph_back = tf.get_default_graph()
        # get the classes dict
        classes_cows_back = util.get_classes_in_keras_format()

        config.model = 'antnet'
        # get the keras original model object
        keras_module_antnet = util.get_model_class_instance()
        # load the best trained model
        keras_model_antnet = keras_module_antnet.load()
        graph_antnet = tf.get_default_graph()
        # get the classes dict
        classes_cows_antnet = util.get_classes_in_keras_format()

        # warm up the model
        print('Warming up the keras model face')
        input_shape = (1,) + keras_module_face.img_size + (3,)
        dummpy_img = numpy.ones(input_shape)
        dummpy_img = preprocess_input(dummpy_img)
        keras_model_face.predict(dummpy_img)

        # warm up the model
        print('Warming up the keras model back')
        input_shape = (1,) + keras_module_back.img_size + (3,)
        dummpy_img = numpy.ones(input_shape)
        dummpy_img = preprocess_input(dummpy_img)
        keras_model_back.predict(dummpy_img)

        # warm up the model
        print('Warming up the keras model antnet')
        input_shape = (1,) + keras_module_antnet.img_size + (3,)
        dummpy_img = numpy.ones(input_shape)
        dummpy_img = preprocess_input(dummpy_img)
        keras_model_antnet.predict([dummpy_img,np.random.rand(1)])

    # load the keras model 
    # app.run()
    # lock = threading.Lock()
    # lock.acquire()
    # lock.release()
    def predict_obj(self, image):
        global sess,net,graph_objdetect
        with graph_objdetect.as_default():
            _, all_coordinates = demo.demo(sess, net, image)
        return all_coordinates

    def predict_face(self, image, all_coordinates):
        global keras_model_face,classes_cows_face,graph_face
        with graph_face.as_default():
            return demo.keras_id_predict(image, all_coordinates, keras_model_face,classes_cows_face)

    def predict_back(self, image, all_coordinates_trans):
        global keras_model_back, classes_cows_back, graph_back
        with graph_back.as_default():
            return demo.keras_id_predict(image, all_coordinates_trans, keras_model_back, classes_cows_back, need_id=True)

    def predict_antnet(self, image):
        global keras_model_antnet, classes_cows_antnet, graph_antnet
        with graph_antnet.as_default():
            return demo.keras_classify_predict(image, keras_model_antnet, classes_cows_antnet)
