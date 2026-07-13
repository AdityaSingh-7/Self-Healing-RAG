import Link from "next/link";
import { FileText, MessageSquare, Upload, Activity, Shield, BarChart3 } from "lucide-react";

export default function HomePage() {
  return (
    <div className="flex items-center justify-center min-h-[calc(100vh-4rem)]">
      <div className="text-center max-w-3xl px-4">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm font-medium mb-4">
          <Shield className="w-4 h-4" />
          Self-Healing Enabled
        </div>

        <h1 className="text-4xl font-bold mb-4">
          Self-Healing RAG System
        </h1>
        <p className="text-lg text-gray-600 mb-8">
          Upload documents, ask questions, and watch the system validate, heal, and improve its own answers in real-time.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <div className="p-6 bg-white rounded-xl shadow-sm border">
            <Upload className="w-8 h-8 text-blue-600 mx-auto mb-3" />
            <h3 className="font-semibold mb-1">Upload</h3>
            <p className="text-sm text-gray-500">
              Drag & drop PDFs, DOCX, TXT, or Markdown
            </p>
          </div>
          <div className="p-6 bg-white rounded-xl shadow-sm border">
            <Shield className="w-8 h-8 text-green-600 mx-auto mb-3" />
            <h3 className="font-semibold mb-1">Self-Heal</h3>
            <p className="text-sm text-gray-500">
              Validates answers and retries with adaptive strategies
            </p>
          </div>
          <div className="p-6 bg-white rounded-xl shadow-sm border">
            <BarChart3 className="w-8 h-8 text-purple-600 mx-auto mb-3" />
            <h3 className="font-semibold mb-1">Measure</h3>
            <p className="text-sm text-gray-500">
              Track confidence, strategies, costs, and accuracy
            </p>
          </div>
        </div>

        <div className="flex gap-4 justify-center flex-wrap">
          <Link
            href="/upload"
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition font-medium"
          >
            Upload Documents
          </Link>
          <Link
            href="/chat"
            className="px-6 py-3 bg-gray-900 text-white rounded-lg hover:bg-gray-800 transition font-medium"
          >
            Ask with Healing
          </Link>
          <Link
            href="/dashboard"
            className="px-6 py-3 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition font-medium"
          >
            View Dashboard
          </Link>
        </div>
      </div>
    </div>
  );
}
