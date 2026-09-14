# Real-Time License Plate Recognition

Bu proje, araç plakalarını gerçek zamanlı olarak tespit edip okumak için geliştirdiğim bir görüntü işleme projesidir.

Sistemde plakanın yerini bulmak için **YOLO**, plaka üzerindeki karakterleri okumak için **EasyOCR** ve görüntü işlemleri için **OpenCV** kullanılmıştır.

## Projede Neler Var?

- YOLO ile plaka tespiti
- OpenCV ile plaka bölgesini kırpma
- Görüntüyü büyütme ve grayscale işlemleri
- EasyOCR ile plaka karakterlerini okuma
- Regex ile OCR sonucunu temizleme
- Aynı plakayı tekrar tekrar kaydetmemek için `set`
- OCR hatalarını azaltmak için son birkaç sonucu kullanan majority voting sistemi
- Tespit edilen plakaları tarih ve saat bilgisiyle `.txt` dosyasına kaydetme
- FPS gösterimi

## Kullanılan Teknolojiler

- Python
- YOLO / Ultralytics
- OpenCV
- EasyOCR
- NumPy
- PyTorch
- Regex
- Collections (`deque`, `Counter`)

## Çalışma Mantığı

Sistem kabaca şu şekilde çalışır:

Video / Kamera  
↓  
YOLO ile plaka tespiti  
↓  
Plaka bölgesini kırpma  
↓  
OpenCV ile görüntü iyileştirme  
↓  
EasyOCR ile karakter okuma  
↓  
OCR sonuçlarını temizleme  
↓  
Majority Voting  
↓  
Plakayı TXT dosyasına kaydetme  

Majority voting sistemi, EasyOCR'ın bazı karelerde örneğin `8` karakterini `B` veya `0` karakterini `O` olarak okuması gibi hataları azaltmak için kullanılmıştır.

## Model

Eğittiğim YOLO modelinin ağırlıkları şu klasörde bulunmaktadır:

`weights/best.pt`

## Dataset

Model eğitiminde kullandığım dataset:

**Dataset Link:**  
https://app.roboflow.com/ygts-workspace/plate-axwcb-rz7go/train

Dataset YOLO formatında hazırlanmıştır.

## Kurulum

Gerekli kütüphaneleri yüklemek için:

`pip install ultralytics opencv-python easyocr numpy torch`

## Kullanım

Ana Python dosyasını çalıştır:

`python Plate_Recognition_System.py`

Video yolu veya kamera kaynağı kod içerisinden değiştirilebilir.

Kamera kullanmak için örnek:

`cap = cv2.VideoCapture(0)`

## Çıktı

Tespit edilen plakalar tarih ve saat bilgisiyle şu dosyaya kaydedilir:

`Plate_SAVING.txt`

Örnek çıktı:

26-09-14 20:54:23 61ABC123  
26-09-14 20:55:10 34XYZ456  

## Not

Bu proje öğrenme ve geliştirme amacıyla hazırlanmıştır. OCR doğruluğu görüntü kalitesi, plaka açısı, ışık ve çözünürlük gibi faktörlere bağlı olarak değişebilir.
