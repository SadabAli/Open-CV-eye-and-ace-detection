from flask import Flask , request ,Response , render_template
import cv2

app = Flask(__name__)
camera = cv2.VideoCapture(0)

def Generate_Image():
    while True:
        success , image =camera.read()
        if not success:
            break
        else:
            face_detect  =cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
            eye_detect = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')
            face = face_detect.detectMultiScale(image,1.1,7)
            gry=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

            # drowing the rectangle around each face
            for(x,y,w,h) in face:
                cv2.rectangle(image,(x,y) , (x+w , y+h) , (255,0,0) , 2)
                roi_gray = gry[y:y+h, x:x+w]
                roi_color = image[y:y+h, x:x+w]
                eye = eye_detect.detectMultiScale(roi_gray,1.1,3)
                for(ex,ey,ew,eh) in eye:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            ret , buffer = cv2.imencode('.png' , image)
            image=buffer.tobytes()
            yield (b'--image\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')



@app.route("/")
def display():
    return render_template('indexx.html')

@app.route("/live_detection")
def live_detection():
    return Response(Generate_Image() , mimetype='multipart/x-mixed-replace; boundary=image')

if __name__== "__main__":
    app.run(debug=True)