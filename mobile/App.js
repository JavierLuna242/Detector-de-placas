import React, { useRef, useState } from 'react';
import {
  Alert,
  Image,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import { detectPlate, getApiUrl } from './api';

export default function App() {
  const [apiUrl, setApiUrl] = useState(getApiUrl());
  const [imageUri, setImageUri] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef(null);

  const handleWebSelection = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const previewUrl = URL.createObjectURL(file);
    setSelectedFile(file);
    setImageUri(previewUrl);
    setResult(null);
  };

  const openWebFilePicker = () => {
    fileInputRef.current?.click();
  };

  const pickImage = async () => {
    if (Platform.OS === 'web') {
      openWebFilePicker();
      return;
    }

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
      setSelectedFile(null);
      setImageUri(selected.assets[0].uri);
      setResult(null);
    }
  };

  const takePhoto = async () => {
    if (Platform.OS === 'web') {
      openWebFilePicker();
      return;
    }

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
      setSelectedFile(null);
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
      const payload = await detectPlate(imageUri, selectedFile);
      setResult(payload);
    } catch (error) {
      Alert.alert('Error', error.message || 'No se pudo procesar la imagen.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaProvider>
      <SafeAreaView style={styles.safeArea}>
        <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            capture={Platform.OS === 'web' ? 'environment' : undefined}
            style={styles.webInput}
            onChange={handleWebSelection}
          />

          <View style={styles.header}>
            <Text style={styles.eyebrow}>Detector inteligente</Text>
            <Text style={styles.title}>Reconocimiento de placas</Text>
          </View>

          <View style={styles.card}>
            <Text style={styles.label}>Servidor</Text>
            <TextInput
              value={apiUrl}
              onChangeText={setApiUrl}
              placeholder="http://<IP_EC2>:8080"
              style={styles.input}
              autoCapitalize="none"
              autoCorrect={false}
            />
          </View>

          <View style={styles.actions}>
            <Pressable style={styles.primaryButton} onPress={takePhoto}>
              <Text style={styles.primaryButtonText}>Tomar foto</Text>
            </Pressable>

            <Pressable style={styles.secondaryButton} onPress={pickImage}>
              <Text style={styles.secondaryButtonText}>Galería</Text>
            </Pressable>
          </View>

          {imageUri ? (
            <Image source={{ uri: imageUri }} style={styles.preview} resizeMode="cover" />
          ) : (
            <View style={styles.placeholder}>
              <Text style={styles.placeholderText}>Esperando una imagen...</Text>
            </View>
          )}

          <Pressable
            style={[styles.detectButton, loading && styles.detectButtonDisabled]}
            onPress={handleSubmit}
            disabled={loading}
          >
            <Text style={styles.detectButtonText}>{loading ? 'Procesando...' : 'Detectar placa'}</Text>
          </Pressable>

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
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#eef4ff',
  },
  content: {
    paddingHorizontal: 20,
    paddingTop: 24,
    paddingBottom: 32,
    gap: 16,
  },
  header: {
    gap: 6,
  },
  eyebrow: {
    fontSize: 12,
    letterSpacing: 1.2,
    textTransform: 'uppercase',
    color: '#2563eb',
    fontWeight: '700',
  },
  title: {
    fontSize: 30,
    fontWeight: '800',
    color: '#0f172a',
  },
  card: {
    backgroundColor: '#ffffff',
    borderRadius: 18,
    padding: 16,
    borderWidth: 1,
    borderColor: '#dbeafe',
    shadowColor: '#93c5fd',
    shadowOpacity: 0.12,
    shadowRadius: 10,
    shadowOffset: { width: 0, height: 6 },
    elevation: 3,
  },
  label: {
    fontSize: 13,
    fontWeight: '700',
    color: '#334155',
    marginBottom: 8,
  },
  input: {
    borderWidth: 1,
    borderColor: '#bfdbfe',
    borderRadius: 12,
    paddingHorizontal: 14,
    paddingVertical: 12,
    backgroundColor: '#f8fbff',
    fontSize: 15,
    color: '#0f172a',
  },
  actions: {
    gap: 10,
  },
  primaryButton: {
    backgroundColor: '#2563eb',
    borderRadius: 16,
    paddingVertical: 16,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#2563eb',
    shadowOpacity: 0.24,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 8 },
    elevation: 4,
  },
  primaryButtonText: {
    color: '#ffffff',
    fontSize: 17,
    fontWeight: '700',
  },
  secondaryButton: {
    backgroundColor: '#e0ecff',
    borderRadius: 16,
    paddingVertical: 14,
    alignItems: 'center',
    justifyContent: 'center',
  },
  secondaryButtonText: {
    color: '#1d4ed8',
    fontSize: 15,
    fontWeight: '700',
  },
  preview: {
    width: '100%',
    height: 280,
    borderRadius: 22,
    backgroundColor: '#dfeafc',
  },
  placeholder: {
    height: 280,
    borderRadius: 22,
    backgroundColor: '#dfeafc',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: '#c7d7f8',
  },
  placeholderText: {
    color: '#475569',
    fontSize: 16,
    fontWeight: '600',
  },
  detectButton: {
    backgroundColor: '#0f172a',
    borderRadius: 16,
    paddingVertical: 16,
    alignItems: 'center',
    justifyContent: 'center',
  },
  detectButtonDisabled: {
    opacity: 0.7,
  },
  detectButtonText: {
    color: '#ffffff',
    fontSize: 17,
    fontWeight: '700',
  },
  resultCard: {
    backgroundColor: '#ffffff',
    borderRadius: 18,
    padding: 16,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  resultTitle: {
    fontSize: 18,
    fontWeight: '800',
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
  webInput: {
    display: 'none',
  },
});
