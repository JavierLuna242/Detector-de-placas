# Despliegue en EC2 + Expo Go

Este proyecto ya queda preparado para ejecutarse en un servidor EC2 y conectarse desde una app Expo Go.

## 1) Backend en EC2

1. Conéctate a la instancia EC2:

```bash
ssh -i "tu_clave.pem" ubuntu@<IP_PUBLICA_EC2>
```

2. En la instancia, instala Python y dependencias:

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv libgl1 libglib2.0-0
```

3. Clona o sube el proyecto a la instancia:

```bash
mkdir -p ~/project
cd ~/project
```

Si estás usando Git:

```bash
git clone <URL_DEL_REPO> .
```

O sube los archivos desde tu PC con SCP:

```bash
scp -r . ubuntu@<IP_PUBLICA_EC2>:/home/ubuntu/project
```

4. Crea un entorno virtual e instala dependencias:

```bash
cd ~/project
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

5. Inicia el backend:

```bash
PORT=8080 python app.py
```

O directamente:

```bash
python app.py
```

La API quedará disponible en:

```text
http://<IP_PUBLICA_EC2>:8080/docs
```

6. Verifica que responde:

```bash
curl http://localhost:8080/
```

## 2) Configuración de puertos en EC2

Asegúrate de abrir el puerto 8080 en el grupo de seguridad de la instancia:

- Tipo: Custom TCP
- Puerto: 8080
- Fuente: 0.0.0.0/0

## 3) App Expo Go

La app móvil quedó creada en la carpeta [mobile](mobile).

Edita el valor de la URL del backend en [mobile/app.json](mobile/app.json) o en la variable de entorno:

```json
"extra": {
  "apiUrl": "http://<IP_PUBLICA_EC2>:8080"
}
```

También puedes usar:

```bash
EXPO_PUBLIC_API_URL=http://<IP_PUBLICA_EC2>:8080 npx expo start
```

## 4) Ejecutar la app mobile con Expo Go

Desde tu equipo:

```bash
cd mobile
npm install
npx expo start
```

Luego escanea el código QR con Expo Go desde tu teléfono.

## 5) Endpoints principales

- GET / -> salud del servidor
- POST /predict/ -> recibe una imagen y devuelve la placa detectada
- POST /predict_json/ -> recibe base64 en JSON

## 6) Recomendación importante

Para producción, cambia el CORS en [src/api/main.py](src/api/main.py) desde:

```python
allow_origins=["*"]
```

a una lista controlada como:

```python
allow_origins=["http://localhost:3000", "https://tu-dominio.com"]
```
