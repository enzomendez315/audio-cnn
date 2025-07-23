"use client";

import Link from "next/link";
import { useState } from "react";
import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import { Card, CardContent } from "~/components/ui/card";

interface Prediction {
  category: string;
  confidence: number;
}

interface LayerData {
  shape: number[];
  values: number[][];
}

interface VisualizationData {
  [layerName: string]: LayerData;
}

interface WaveformData {
  values: number[];
  sampleRate: number;
  duration: number;
}

interface ApiResponse {
  prediction: Prediction[];
  visualization: VisualizationData;
  inputSpectogram: LayerData;
  waveform: WaveformData;
}

export default function HomePage() {
  const [visualizationData, setVisualizationData] =
    useState<ApiResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [fileName, setFileName] = useState("");
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];

    if (!file) {
      return;
    }

    setFileName(file.name);
    setIsLoading(true);
    setError(null);
    setVisualizationData(null);

    const reader = new FileReader();
    reader.readAsArrayBuffer(file);
    reader.onload = async () => {
      try {
        // Convert binary data into base64 encoder string
        const arrayBuffer = reader.result as ArrayBuffer;
        const base64String = btoa(
          new Uint8Array(arrayBuffer).reduce(
            (data, byte) => data + String.fromCharCode(byte),
            "",
          ),
        );

        const response = await fetch(
          "https://enzomendez315--audio-cnn-inference-audioclassifier-inference.modal.run",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ audio_data: base64String }),
          },
        );

        if (!response.ok) {
          throw new Error(`API error ${response.statusText}`);
        }

        const data: ApiResponse = await response.json();
        setVisualizationData(data);
      } catch (error) {
        setError(
          error instanceof Error ? error.message : "An uknown error occured",
        );
      } finally {
        setIsLoading(false);
      }
    };

    reader.onerror = () => {
      setError("Failed to read the file");
      setIsLoading(false);
    };
  };

  return (
    <main className="min-h-screen bg-stone-50 p-8">
      <div className="mx-auto max-w-[60%]">
        <div className="mb-12 text-center">
          <h1 className="mb-4 text-4xl font-light tracking-tight text-stone-900">
            CNN Audio Visualizer
          </h1>
          <p className="text-md mb-8 text-stone-600">
            Upload a WAV file to see the model's predictions and feature maps.
          </p>
          <div className="flex flex-col items-center">
            <div className="relative inline-block">
              <input
                type="file"
                accept=".wav"
                id="file-upload"
                onChange={handleFileChange}
                disabled={isLoading}
                className="absolute inset-0 w-full cursor-pointer opacity-0"
              />
              <Button
                disabled={isLoading}
                className="border-stone-300"
                variant="outline"
                size="lg"
              >
                {isLoading ? "Analyzing..." : "Choose File"}
              </Button>
            </div>

            {fileName && (
              <Badge
                variant="secondary"
                className="mt-4 bg-stone-200 text-stone-700"
              >
                {fileName}
              </Badge>
            )}
          </div>
        </div>

        {error && (
          <Card className="mb-8 border-red-200 bg-red-50">
            <CardContent>
              <p className="text-red-600">Error: {error}</p>
            </CardContent>
          </Card>
        )}
      </div>
    </main>
  );
}
