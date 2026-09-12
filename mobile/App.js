import React, { useState } from 'react';
import {
  Alert,
  Button,
  Image,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { SafeAreaView } from 'react-native-safe-area-context';
import Constants from 'expo-constants';
import { detectPlate, getApiUrl } from './api';

export default function App() {
  const [apiUrl, setApiUrl] = useState(getApiUrl());
  const [imageUri, setImageUri] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const pickImage = async () => {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      Alert.alert('Permiso requerido', 'Necesitas permitir acceso a tus fotos.');
      return;
    }

    const selected = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.8,
      allowsEditing: true,
    });

    if (!selected.canceled) {
      setImageUri(selected.assets[0].uri);
      setResult(null);
    }
  };

  const takePhoto = async () => {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      Alert.alert('Permiso requerido', 'Necesitas permitir acceso a la cámara.');
      return;
    }

    const photo = await ImagePicker.launchCameraAsync({
      quality: 0.8,
      allowsEditing: true,
    });

    if (!photo.canceled) {
      setImageUri(photo.assets[0].uri);
      setResult(null);
    }
  };

  const handleSubmit = async () => {
    if (!imageUri) {
      Alert.alert('Falta imagen', 'Selecciona una imagen o toma una foto.');
      return;
    }

    setLoading(true);
    try {
      const payload = await detectPlate(imageUri);
      setResult(payload);
    } catch (error) {
      Alert.alert('Error', error.message || 'No se pudo procesar la imagen.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.title}>Detector de Placas</Text>
        <Text style={styles.subtitle}>App para Expo Go</Text>

        <Text style={styles.label}>URL del backend EC2</Text>
        <TextInput
          value={apiUrl}
          onChangeText={setApiUrl}
          placeholder="http://<IP_EC2>:8080"
          style={styles.input}
        />

        <View style={styles.buttonRow}>
          <Button title="Galería" onPress={pickImage} />
          <Button title="Cámara" onPress={takePhoto} />
        </View>

        {imageUri ? (
          <Image source={{ uri: imageUri }} style={styles.preview} resizeMode="cover" />
        ) : (
          <View style={styles.placeholder}>
            <Text style={styles.placeholderText}>No hay imagen seleccionada</Text>
          </View>
        )}

        <View style={styles.submitButton}>
          <Button
            title={loading ? 'Procesando...' : 'Detectar placa'}
            onPress={handleSubmit}
            disabled={loading}
          />
        </View>

        {result && (
          <View style={styles.resultCard}>
            <Text style={styles.resultTitle}>Resultado</Text>
            <Text style={styles.resultText}>
              {result.success ? `Placas: ${result.placas?.join(', ') || 'No detectadas'}` : 'No se pudo procesar'}
            </Text>
            {result.image && (
              <Image
                source={{ uri: `data:image/jpeg;base64,${result.image}` }}
                style={styles.resultImage}
                resizeMode="contain"
              />
            )}
            <Text style={styles.resultText}>{result.message || 'OK'}</Text>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f7fb',
  },
  content: {
    padding: 20,
    gap: 12,
  },
  title: {
    fontSize: 26,
    fontWeight: '700',
    color: '#0f172a',
  },
  subtitle: {
    fontSize: 16,
    color: '#475569',
    marginBottom: 6,
  },
  label: {
    fontSize: 14,
    fontWeight: '600',
    color: '#334155',
    marginTop: 8,
  },
  input: {
    borderWidth: 1,
    borderColor: '#cbd5e1',
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 10,
    backgroundColor: '#fff',
    fontSize: 15,
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 12,
  },
  preview: {
    width: '100%',
    height: 260,
    borderRadius: 16,
    marginTop: 8,
  },
  placeholder: {
    height: 220,
    borderRadius: 16,
    backgroundColor: '#e2e8f0',
    alignItems: 'center',
    justifyContent: 'center',
  },
  placeholderText: {
    color: '#475569',
    fontSize: 14,
  },
  submitButton: {
    marginTop: 4,
  },
  resultCard: {
    backgroundColor: '#ffffff',
    borderRadius: 16,
    padding: 14,
    borderWidth: 1,
    borderColor: '#e2e8f0',
    marginTop: 10,
  },
  resultTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#0f172a',
    marginBottom: 8,
  },
  resultText: {
    fontSize: 14,
    color: '#334155',
    marginBottom: 10,
  },
  resultImage: {
    width: '100%',
    height: 220,
    borderRadius: 12,
    marginBottom: 10,
  },
});
