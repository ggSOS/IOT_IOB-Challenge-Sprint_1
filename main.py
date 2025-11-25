import cv2, dlib, numpy as np, pickle, os, time


PREDICTOR = "IA_models/shape_predictor_5_face_landmarks.dat"
RECOG = "IA_models/dlib_face_recognition_resnet_model_v1.dat"
DB_FILE = "database/db.pkl"
THRESH = 0.5 ## menor = mais precisão de identificacao


print("Configurando Ambiente...")
db = pickle.load(open(DB_FILE,"rb")) if os.path.exists(DB_FILE) else {}
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(PREDICTOR)
rec = dlib.face_recognition_model_v1(RECOG)
time.sleep(2)


def window_size_adjuster(width, height):
    if width>1200 or height>700:
        width/=2
        height/=2
        width, height = window_size_adjuster(width, height)
    return round(width), round(height)


## unificar configurações dos textos
def text_constructor(x_coordinates, y_coordinate, main_text, warn_text=None, status_ok=False, ):
    if warn_text:
        if status_ok:
            text_message = main_text
            text_font_color = (0, 123, 0)
        else:
            text_message = warn_text
            text_font_color = (0, 0, 123)
    else:
        text_message = main_text
        text_font_color = (123, 0, 0)
    text_coordinates = (x_coordinates, y_coordinate)
    text_font_family = cv2.FONT_HERSHEY_SIMPLEX
    text_font_size = 1.5
    text_font_weight = 4
    text_line_draw = cv2.LINE_AA ## Anti-aliasing
    return text_message, text_coordinates, text_font_family, text_font_size, text_font_color, text_font_weight, text_line_draw


def door_new_status(door_opened, door_opening_allowed, time_last_opening, time_now, door_cooldown):
    if door_opening_allowed:
        if not door_opened:
            print("- Aberta")
        return True, time_now    
    else:
        if door_opened:
            if time_now-time_last_opening >= door_cooldown:
                print("- Fechada")
            else:
                return True, time_last_opening
        return False, time_last_opening
def door_status_print(door_opened):
    if door_opened:
        print("\nStatus da porta:\n- Aberta")
    else:
        print("\nStatus da porta:\n- Fechada")


video_path = "assets/videos/small_ver_no_copyright_faces.MOV"
cap = cv2.VideoCapture(video_path)
original_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
new_frame_width, new_frame_height = window_size_adjuster(original_frame_width, original_frame_height)

escaneando = False
ultimo = 0
door_cooldown = 3
door_opened = False
door_opening_allowed = False
time_last_opening = time.time()

print("Comandos:\n\t[1] = Cadastrar\n\t[2] = Escanear ON/OFF\n\t[3] = Sair")
door_status_print(door_opened)
while True:
    ok, frame = cap.read()
    if not ok: break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rects = detector(rgb, 0) ## maior para mais zoom/precisão, mas fica mais pesado
    door_opening_allowed = False

    for r in rects:
        shape = sp(rgb, r)
        chip = dlib.get_face_chip(rgb, shape)
        vec = np.array(rec.compute_face_descriptor(chip), dtype=np.float32)

        if escaneando and db:
            nome, dist = "Desconhecido", 999
            for n, (v) in db.items():
                d = np.linalg.norm(vec - v)
                if d < dist:
                    nome, dist = n, d
            if dist > THRESH:
                nome = "Desconhecido"
            else:
                door_opening_allowed = True
            frase_frame = f"{nome}"

            color = (123,0,0) if nome != "Desconhecido" else (0,0,123)
            cv2.rectangle(frame, (r.left(), r.top()), (r.right(), r.bottom()), color, 4)
            cv2.putText(frame, *text_constructor(r.left(), r.top()-10, frase_frame))
    door_opened, time_last_opening = door_new_status(door_opened, door_opening_allowed, time_last_opening, time.time(), door_cooldown)


    ## Alertas de status
    frame = cv2.rectangle(frame, (10, 10), (550, 120), (0,0,0), -1)
    frame = cv2.putText(frame, *text_constructor(20, 50, ">Database Econtrado", ">Database Inexistente", db))
    frame = cv2.putText(frame, *text_constructor(20, 100, ">Scanner ON", ">Scanner OFF", escaneando))

    ## Alertas de comandos
    frame = cv2.putText(frame, *text_constructor(10, new_frame_height-100, "[1] = Cadastrar"))
    frame = cv2.putText(frame, *text_constructor(10, new_frame_height-50, "[2] = Escanear ON/OFF"))
    frame = cv2.putText(frame, *text_constructor(10, new_frame_height, "[3] = Sair"))


    cv2.namedWindow("Faces", cv2.WINDOW_NORMAL)
    cv2.imshow("Faces", frame)
    cv2.resizeWindow("Faces", new_frame_width, new_frame_height)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('3'): break
    if k == ord('2'): escaneando = not escaneando
    if k == ord('1') and len(rects) == 1:
        nome = input("\nNome para Cadastro:\n\t- ").strip()
        while not nome:
            print("\nNome Inválido!\n")
            nome = input("Nome para Cadastro:\n\t- ").strip()
        
        db[nome] = (vec)
        pickle.dump(db, open(DB_FILE,"wb"))
        print(f"Salvo: {nome}")
        door_status_print(door_opened)
            

cap.release()
cv2.destroyAllWindows()
