'''
    - 필요 패키지 설치
    - pip install tensorflow
    - pip install transformers
        - 맥북 계열은 아래 방식 참고
        - https://jamescalam.medium.com/hugging-face-and-sentence-transformers-on-m1-macs-4b12e40c21ce
    - pip install flask-socketio
    - conda install pytorch
    
'''
from flask import Flask, render_template
import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFGPT2LMHeadModel
import numpy as np
from flask_socketio import SocketIO, emit
import random

app = Flask(__name__)
# 실시간 통신(SocketIO 사용)을 위해 비밀키 지정
app.config['SECRET_KEY'] = 'dfghjk9084567uplkj' # 임의의 키값을 지정
socketio = SocketIO(app)

# repo      = 'skt/kogpt2-base-v2'                    # 허깅페이스에 등록된 사전 학습된 모델의 레포지토리명
# tokenizer = AutoTokenizer.from_pretrained( repo )   # 토크나이저로드
# model     = TFGPT2LMHeadModel.from_pretrained( repo, from_pt=True )  # 사전 학습된 모델(가중치 포함) 로드


@app.route('/')
def home():
    return render_template('index_.html')

@app.route('/nlp/create_sentence')
def create_sentence():
    return render_template('pages/create_sentence.html')

# 클라이언트 메시지를 처리하기 위한 이벤트 등록및 처리
@socketio.on('cTos_simple_msg')
def cTos_simple_msg( data ):
    print( data )
    # echo, 받는 내용을 살짝 수정해서 응답(서버 => 클라이언트로 메시지 전송, 푸시)
    data['msg'] += "<응답" 
    emit('sToc_simple_msg', data )

@socketio.on('cTos_create_sentence')
def cTos_create_sentence( data ):
    seed_sentence = data['seed']
    print( '문장생성의 재료 문장 획득', seed_sentence )
    # 1. 문장 -> 인코딩(백터화)
    input_vectors = tokenizer.encode( seed_sentence )
    # 2. 제한된 문장 길이만큼 다음 토큰을 예측해서 클라이언트로 푸시
    while len(input_vectors) < 50:
        output        = model( np.array([ input_vectors ]) )
        top5          = tf.math.top_k( output.logits[0, -1], k=5)
        token_id      = random.choice(  top5.indices.numpy() )
        input_vectors.append( token_id )
        # 클라이언트에게 예측 토큰 전송
        emit('sToc_create_sentence', { 'word':tokenizer.decode( [token_id] )} )
    

if __name__ == '__main__':
    #app.run()
    # 소켓IO를 지원하는 기반 서버 가동
    socketio.run(app, port=3333, debug=True)
