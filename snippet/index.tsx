import { CameraView, useCameraPermissions } from 'expo-camera';
import * as Speech from 'expo-speech';
import React, { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Button,
  Image,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

export default function CameraScreen() {
  const cameraRef = useRef<any>(null);
  const [permission, requestPermission] = useCameraPermissions();
  const [ip, setIp] = useState('');
  const [port, setPort] = useState('8080');
  const [image, setImage] = useState<string | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [plates, setPlates] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  const apiUrl = ip && port ? `http://${ip}:${port}` : '';

  useEffect(() => {
    if (!permission) {
      requestPermission();
    }
  }, [permission]);

  const handleCapture = async () => {
    if (!cameraRef.current) return;
    if (!ip) {
      Alert.alert('Error', 'Por favor ingresa la dirección IP del servidor.');
      return;
    }

    try {
      setLoading(true);
      const photo = await cameraRef.current.takePictureAsync({ base64: true });
      setImage(photo.uri);
      setPlates([]);
      setProcessedImage(null);

      const fullUrl = `${apiUrl.endsWith('/') ? apiUrl.slice(0, -1) : apiUrl}/predict/`;
      console.log('📤 Enviando imagen base64 a:', fullUrl);

      const formBody = new URLSearchParams();
      formBody.append('image_base64', photo.base64);

      const response = await fetch(fullUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
          Accept: 'application/json',
        },
        body: formBody.toString(),
      });

      if (!response.ok) {
        const text = await response.text();
        console.error('❌ Error HTTP:', response.status, text);
        Alert.alert('Error HTTP', `Código: ${response.status}`);
        Speech.speak('Ocurrió un error al contactar el servidor.');
        return;
      }

      const data = await response.json();
      console.log('📥 Respuesta del servidor:', data);

      if (data?.placas && data.placas.length > 0) {
        const detected = data.placas;
        setPlates(detected);

        if (data.image) {
          setProcessedImage(`data:image/jpeg;base64,${data.image}`);
        }

        const textToSpeak =
          detected.length === 1
            ? `La placa detectada es ${detected[0].split('').join(' ')}`
            : `Se detectaron ${detected.length} placas: ${detected.join(', ')}`;

        if (Platform.OS !== 'web') {
          Speech.speak(textToSpeak, { language: 'es-ES' });
        }
      } else if (data?.placas?.length === 0) {
        if (Platform.OS !== 'web') Speech.speak('No se detectaron placas.');
        Alert.alert('Resultado', 'No se detectaron placas.');
        setPlates([]);
        setProcessedImage(null);
      } else if (data?.error) {
        Alert.alert('Error del servidor', data.error);
        if (Platform.OS !== 'web') Speech.speak('Ocurrió un error en el servidor.');
      } else {
        console.warn('⚠️ Respuesta inesperada:', data);
        Alert.alert('Respuesta inesperada', JSON.stringify(data));
      }
    } catch (error) {
      console.error('❌ Error enviando imagen:', error);
      Alert.alert('Error', 'No se pudo conectar al servidor.');
      if (Platform.OS !== 'web') Speech.speak('No se pudo conectar al servidor.');
    } finally {
      setLoading(false);
    }
  };

  if (!permission) {
    return (
      <View style={styles.container}>
        <Text>Solicitando permisos...</Text>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text>Se necesita permiso para usar la cámara.</Text>
        <Button title="Conceder permiso" onPress={requestPermission} />
      </View>
    );
  }

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.label}>Dirección IP del servidor:</Text>
      <TextInput
        style={styles.input}
        placeholder="Ej: 192.168.1.45"
        value={ip}
        onChangeText={setIp}
      />

      <Text style={styles.label}>Puerto:</Text>
      <TextInput
        style={styles.input}
        placeholder="8080"
        value={port}
        onChangeText={setPort}
        keyboardType="numeric"
      />

      <CameraView ref={cameraRef} style={styles.camera} facing="back" />
      <Button title="Tomar foto" onPress={handleCapture} color="#007AFF" />

      {loading && <ActivityIndicator size="large" color="#007AFF" style={{ marginTop: 20 }} />}

      {image && (
        <View style={styles.imageContainer}>
          <Text style={styles.label}>📷 Imagen capturada:</Text>
          <Image source={{ uri: image }} style={styles.image} />
        </View>
      )}

      {processedImage && (
        <View style={styles.imageContainer}>
          <Text style={styles.label}>🖼️ Imagen procesada por el servidor:</Text>
          <Image source={{ uri: processedImage }} style={styles.image} resizeMode="contain" />
        </View>
      )}

      {plates.length > 0 && (
        <View style={styles.resultContainer}>
          <Text style={styles.label}>🚘 Placas detectadas:</Text>
          {plates.map((p, i) => (
            <Text key={i} style={styles.plateText}>{p}</Text>
          ))}
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    alignItems: 'center',
    justifyContent: 'flex-start',
    backgroundColor: '#f5f5f5',
    padding: 16,
  },
  label: {
    fontWeight: 'bold',
    marginBottom: 6,
    color: '#333',
  },
  input: {
    width: '90%',
    height: 40,
    borderColor: '#ccc',
    borderWidth: 1,
    borderRadius: 8,
    paddingHorizontal: 10,
    marginBottom: 10,
    backgroundColor: '#fff',
  },
  camera: {
    width: '100%',
    height: 400,
    borderRadius: 12,
    overflow: 'hidden',
    marginBottom: 16,
  },
  imageContainer: {
    marginTop: 16,
    alignItems: 'center',
  },
  image: {
    width: 300,
    height: 200,
    borderRadius: 10,
  },
  resultContainer: {
    marginTop: 20,
    backgroundColor: '#007AFF20',
    padding: 12,
    borderRadius: 8,
    width: '90%',
  },
  plateText: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#007AFF',
    textAlign: 'center',
  },
});
