import Link from "next/link";
import { FileText, MessageSquare, Upload } from "lucide-react";

export default function HomePage() {
  return (
    <div className="flex items-center justify-center min-h-[calc(100vh-4rem)]">
      <div className="text-center max-w-2xl px-4">
        <h1 className="text-4xl font-bold mb-4">
          Chat with your Documents
        </h1>
        <p className="text-lg text-gray-600 mb-8">
          Upload PDFs, ask questions, get answers with source citations.
          Powered by semantic search, recency-aware retrieval, and Llama 3.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <div className="p-6 bg-white rounded-xl shadow-sm border">
            <Upload className="w-8 h-8 text-blue-600 mx-auto mb-3" />
            <h3 className="font-semibold mb-1">Upload</h3>
            <p className="text-sm text-gray-500">
              Drag & drop PDFs to ingest into the system
            </p>
          </div>
          <div className="p-6 bg-white rounded-xl shadow-sm border">
            <MessageSquare className="w-8 h-8 text-green-600 mx-auto mb-3" />
            <h3 className="font-semibold mb-1">Ask</h3>
            <p className="text-sm text-gray-500">
              Ask questions in natural language
            </p>
          </div>
          <div className="p-6 bg-white rounded-xl shadow-sm border">
            <FileText className="w-8 h-8 text-purple-600 mx-auto mb-3" />
            <h3 className="font-semibold mb-1">Cite</h3>
            <p className="text-sm text-gray-500">
              Get answers with exact source citations
            </p>
          </div>
        </div>

        <div className="flex gap-4 justify-center">
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
            Start Chatting
          </Link>
        </div>
      </div>
    </div>
  );
}
