## Finger Picker App 

- Bu proje, **MediaPipe** ve **OpenCV** kütüphaneleri kullanılarak gerçek zamanlı parmak hareketlerini algılayan ve takip ederek toplama işlemi yapmamızı sağlayan bir uygulamadır.Potansiyel olarak parmak adedini saymak veya etkileşimli kontrol sistemleri oluşturmak için kullanılabilir.

## ✨ Özellikler
 - Gerçek zamanlı kamera akışı izleme
 - MediaPipe Hands modeli kullanılarak **el ve parmak hareketleri ile parmak sayma işlemi**
 - Parmak hareketlerinin analizi için gerekli kütüphanelerin kullanımı


## ⚙️ Kurulum

Uygulamayı yerel makinenizde çalıştırmak için aşağıdaki adımları takip edin.

### Ön Koşullar
 
Sisteminizde kurulu olması gerekenler:

* Python 3.10 sürümü
* Kamera erişim izninin açık olması(webcam) 

### Bağımlılıkların Yüklenmesi

Gerekli tüm Python kütüphanelerini yüklemek için aşağıdaki komutu kullanın:

```bash
pip install opencv-python mediapipe
