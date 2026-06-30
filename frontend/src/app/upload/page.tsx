import { UploadDropzone } from "@/components/UploadDropzone";

export default function UploadPage() {
  return (
    <div className="max-w-3xl mx-auto p-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold">Upload Documents</h1>
        <p className="text-gray-600 mt-2">
          Upload PDF files to ingest into the RAG system. Once processed, you
          can ask questions about them in the chat.
        </p>
      </div>
      <UploadDropzone />
    </div>
  );
}
