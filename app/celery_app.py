# celery_app.py
from celery import Celery, shared_task
from celery.signals import worker_process_init
import base64
import os
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import torch
from sentence_transformers import SentenceTransformer

celery_app = Celery('vectorizer',
#                     broker='amqp://admin:admin@35.206.237.189:5672//',
                    broker='redis://35.206.237.189:5672/0',
                    backend='redis://35.206.237.189:5672/1')


@worker_process_init.connect
def env_settings(*args, **kwargs):
    # SSAFY에서 받은 GPU 사용하도록 설정
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    

def serialize_ndarray(array, shape, dtype):
    bytes_data = array.tobytes()
    base64_encoded = base64.b64encode(bytes_data).decode('utf-8')
    # JSON Response를 위해 직렬화 후 반환
    return {'data': base64_encoded, 'shape': shape, 'dtype': dtype}

    
@shared_task(name="vector_embedding")
def vector_embedding(text):
    # 모델 초기화
    nltk.download('punkt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    model = model.to(device)
    print('Model loaded. Device:', device)
    
    sentences = sent_tokenize(text)
    sentence_embeddings = model.encode(sentences)
    sentence_embeddings = [torch.tensor(embedding) for embedding in sentence_embeddings]
    document_embedding = torch.mean(torch.stack(sentence_embeddings), dim=0)
    
    serialized_data = {
        'data': document_embedding.tolist(),  # 리스트 데이터
        'shape': list(document_embedding.shape),  # 텐서의 모양
        'dtype': str(document_embedding.dtype)  # 데이터 타입
    }
    
    return serialized_data


def main():
    celery_app.worker_main(argv=['worker', '--loglevel=WARNING', '--concurrency=1'])

if __name__ == '__main__':
    main()
