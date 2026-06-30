"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, FileText, CheckCircle, XCircle, Loader2 } from "lucide-react";
import { uploadDocument } from "@/lib/api";

interface UploadResult {
  filename: string;
  status: "success" | "error";
  message: string;
  chunks?: number;
  pages?: number;
}

export function UploadDropzone() {
  const [results, setResults] = useState<UploadResult[]>([]);
  const [isUploading, setIsUploading] = useState(false);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    setIsUploading(true);

    for (const file of acceptedFiles) {
      try {
        const response = await uploadDocument(file);
        setResults((prev) => [
          {
            filename: file.name,
            status: "success",
            message: response.message,
            chunks: response.chunks_created,
            pages: response.pages_parsed,
          },
          ...prev,
        ]);
      } catch (err: any) {
        setResults((prev) => [
          {
            filename: file.name,
            status: "error",
            message: err.message || "Upload failed",
          },
          ...prev,
        ]);
      }
    }

    setIsUploading(false);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
    maxSize: 20 * 1024 * 1024, // 20MB
  });

  return (
    <div className="space-y-6">
      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition ${
          isDragActive
            ? "border-blue-500 bg-blue-50"
            : "border-gray-300 hover:border-blue-400 hover:bg-gray-50"
        }`}
      >
        <input {...getInputProps()} />
        {isUploading ? (
          <Loader2 className="w-12 h-12 text-blue-500 mx-auto animate-spin" />
        ) : (
          <Upload className="w-12 h-12 text-gray-400 mx-auto" />
        )}
        <p className="mt-4 text-lg font-medium text-gray-700">
          {isDragActive
            ? "Drop your PDFs here..."
            : "Drag & drop PDF files here"}
        </p>
        <p className="mt-1 text-sm text-gray-500">
          or click to browse. Max 20MB per file.
        </p>
      </div>

      {/* Upload results */}
      {results.length > 0 && (
        <div className="space-y-3">
          <h3 className="font-semibold text-gray-700">Upload History</h3>
          {results.map((result, i) => (
            <div
              key={i}
              className={`flex items-start gap-3 p-4 rounded-lg border ${
                result.status === "success"
                  ? "bg-green-50 border-green-200"
                  : "bg-red-50 border-red-200"
              }`}
            >
              {result.status === "success" ? (
                <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
              ) : (
                <XCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
              )}
              <div>
                <div className="flex items-center gap-2">
                  <FileText className="w-4 h-4 text-gray-500" />
                  <span className="font-medium text-sm">{result.filename}</span>
                </div>
                <p className="text-sm text-gray-600 mt-1">{result.message}</p>
                {result.pages && (
                  <p className="text-xs text-gray-400 mt-1">
                    {result.pages} pages → {result.chunks} chunks
                  </p>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
