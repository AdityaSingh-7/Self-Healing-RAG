"use client";

import { useState } from "react";
import { X, ChevronLeft, ChevronRight, Highlighter } from "lucide-react";

interface PdfViewerProps {
  docId: string;
  filename: string;
  page: number;
  highlightText?: string;
  onClose: () => void;
}

/**
 * PDF Annotation Viewer
 *
 * Shows the original PDF with highlighted cited text.
 * Uses the /ingest/preview endpoint to fetch the PDF
 * and an iframe to display it with page navigation.
 */
export function PdfViewer({ docId, filename, page, highlightText, onClose }: PdfViewerProps) {
  const [currentPage, setCurrentPage] = useState(page);

  // Build the PDF URL — the backend serves the file
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
  const pdfUrl = `${backendUrl}/ingest/preview/${docId}/${filename}#page=${currentPage}`;

  return (
    <div className="fixed inset-0 z-50 flex">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />

      {/* Viewer panel */}
      <div className="relative ml-auto w-full max-w-3xl bg-white dark:bg-gray-900 shadow-2xl flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b dark:border-gray-700">
          <div className="flex items-center gap-3">
            <h3 className="font-semibold text-sm dark:text-white">{filename}</h3>
            <span className="text-xs text-gray-500">Page {currentPage}</span>
          </div>

          <div className="flex items-center gap-2">
            {/* Page navigation */}
            <button
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              className="p-1 hover:bg-gray-100 dark:hover:bg-gray-800 rounded"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <span className="text-sm text-gray-600 dark:text-gray-400 min-w-[3ch] text-center">
              {currentPage}
            </span>
            <button
              onClick={() => setCurrentPage((p) => p + 1)}
              className="p-1 hover:bg-gray-100 dark:hover:bg-gray-800 rounded"
            >
              <ChevronRight className="w-4 h-4" />
            </button>

            {/* Close */}
            <button
              onClick={onClose}
              className="p-1 hover:bg-gray-100 dark:hover:bg-gray-800 rounded ml-4"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Highlighted text callout */}
        {highlightText && (
          <div className="px-4 py-2 bg-yellow-50 dark:bg-yellow-900/20 border-b border-yellow-200 dark:border-yellow-800">
            <div className="flex items-center gap-2">
              <Highlighter className="w-4 h-4 text-yellow-600" />
              <span className="text-xs font-medium text-yellow-800 dark:text-yellow-200">
                Cited text:
              </span>
            </div>
            <p className="text-xs text-yellow-700 dark:text-yellow-300 mt-1 line-clamp-3">
              &ldquo;{highlightText}&rdquo;
            </p>
          </div>
        )}

        {/* PDF iframe */}
        <div className="flex-1">
          <iframe
            src={pdfUrl}
            className="w-full h-full border-0"
            title={`${filename} - Page ${currentPage}`}
          />
        </div>
      </div>
    </div>
  );
}
