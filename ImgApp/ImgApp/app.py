from werkzeug.utils import secure_filename
from flask import Flask, app, render_template, request
import cv2
import os
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__,static_folder="static") # 静的ファイルを格納するディレクトリをstaticに指定

@app.route('/', methods=["GET", "POST"])
def index():
    # トップページ
    return render_template("index.html")

# ユーザーの入力に対して返答を返す
@app.route('/result', methods=['POST'])
def getResult():
    imgDict = {} 
    print(BASE_DIR)
    #  cv2.BFMatcher:画像から得られた特徴量記述子の距離(ここではハミング距離)を総当たりで計算し、最も近いものをマッチング
    # cv2.NORM_HAMMING:ハミング距離で距離による算出
    bf = cv2.BFMatcher(cv2.NORM_HAMMING) 
    detector = cv2.AKAZE_create() #AKAZE 特徴量検出器を作成
    if request.method == "GET":
        return render_template("index.html")
    else:
        imgName = request.files['inputImg'].filename # ファイル名取得
        if request.files["inputImg"]:
            imgFile = request.files['inputImg'] # 画像ファイルとして読み込む
            radioBtn = request.form['category'] # ラジオボタンを取得
            img_array = np.asarray(bytearray(imgFile.stream.read()), dtype=np.uint8)
            tarImg = cv2.imdecode(img_array,  cv2.IMREAD_GRAYSCALE) # カラー画像として読み込む
            saveImg = cv2.imdecode(img_array,  cv2.IMREAD_COLOR) # カラー画像として読み込む
            target = cv2.resize(tarImg, (400, 400)) # 画像サイズ変更
            if radioBtn == "fashion":
                UPLOAD_FOLDER = BASE_DIR / "static" / "fashionImg"
                app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER # アップロードされたファイルを格納する場所を指定 
                resultPath = "static\\fashionImg\\" #ファイルパスを設定
                IMG_DIR = os.path.abspath(os.path.dirname(__file__)) + '\\static\\fashionImg\\'
                cv2.imwrite(str(BASE_DIR) + "\\static\\fashionImg\\"+ imgName, saveImg) # アップロードした画像の保存
            elif radioBtn == "other":
                UPLOAD_FOLDER = BASE_DIR / "static" / "other"
                app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER # アップロードされたファイルを格納する場所を指定 
                resultPath = "static\\other\\" #ファイルパスを設定
                IMG_DIR = os.path.abspath(os.path.dirname(__file__)) + '\\static\\ohter\\'
                cv2.imwrite(str(BASE_DIR) + "\\static\\other\\"+ imgName, saveImg) # アップロードした画像の保存
            elif radioBtn == "food":
                UPLOAD_FOLDER = BASE_DIR / "static" / "foodImg"
                app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER # アップロードされたファイルを格納する場所を指定 
                resultPath = "static\\foodImg\\" #ファイルパスを設定
                IMG_DIR = os.path.abspath(os.path.dirname(__file__)) + '\\static\\foodImg\\'
                cv2.imwrite(str(BASE_DIR) + "\\static\\foodImg\\"+ imgName, saveImg) # アップロードした画像の保存

            (target_kp, target_des) = detector.detectAndCompute(target, None) # 特徴点の検出と特徴量記述子の計算を一度に行う
            files = os.listdir(IMG_DIR) # 指定フォルダ内のファイルを全てを取得
            for file in files:
                if file == '.DS_Store' or file == imgName:
                    continue
                comparing_img_path = IMG_DIR + file
                try:
                    comparing_img = cv2.imread(comparing_img_path, cv2.IMREAD_COLOR) # カラー画像として読み込む
                    comparing_img = cv2.resize(comparing_img, (200, 200)) # 画像サイズ変更
                    (comparing_kp, comparing_des) = detector.detectAndCompute(comparing_img, None) # 特徴点の検出と特徴量記述子の計算を一度に行う
                    matches = bf.match(target_des, comparing_des)
                    dist = [m.distance for m in matches] # .distance:特徴量記述子間の距離
                    ret = sum(dist) / len(dist)
                except cv2.error:
                    print("Error")
                    ret = 100000
                imgDict[file] = ret
            sortDict = sorted(imgDict.items(), key=lambda x:x[1]) # 昇順ソート
            # sortDict = sorted(imgDict.items(), key=lambda x:x[1], reverse=True) # 昇順
            resultImg = [resultPath + p[0] for p in sortDict[:5]] # ["static\\images\\" + p[0] for p in sortDict[:5]]:画像ファイルパスをリストとして格納する
            print(resultPath+imgName)
        return render_template("index.html", images=resultImg, uploadImg=resultPath+imgName)

if __name__ == '__main__':
    app.run(debug=True, host="localhost")