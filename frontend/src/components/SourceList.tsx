"use client";

import { useState } from "react";
import { GripVertical, Pin, EyeOff, ExternalLink } from "lucide-react";

interface Source {
  text: string;
  filename: string;
  page: number;
  score: number;
  recency_score: number;
  final_score: number;
}

interface SourceListProps {
  sources: Source[];
  onViewPdf?: (filename: string, page: number) => void;
}

export function SourceList({ sources, onViewPdf }: SourceListProps) {
  const [items, setItems] = useState(sources);
  const [pinned, setPinned] = useState<Set<number>>(new Set());
  const [hidden, setHidden] = useState<Set<number>>(new Set());
  const [dragIndex, setDragIndex] = useState<number | null>(null);

  const handleDragStart = (index: number) => {
    setDragIndex(index);
  };

  const handleDragOver = (e: React.DragEvent, index: number) => {
    e.preventDefault();
    if (dragIndex === null || dragIndex === index) return;

    const newItems = [...items];
    const draggedItem = newItems[dragIndex];
    newItems.splice(dragIndex, 1);
    newItems.splice(index, 0, draggedItem);
    setItems(newItems);
    setDragIndex(index);
  };

  const handleDragEnd = () => {
    setDragIndex(null);
  };

  const togglePin = (index: number) => {
    const newPinned = new Set(pinned);
    if (newPinned.has(index)) {
      newPinned.delete(index);
    } else {
      newPinned.add(index);
    }
    setPinned(newPinned);
  };

  const toggleHidden = (index: number) => {
    const newHidden = new Set(hidden);
    if (newHidden.has(index)) {
      newHidden.delete(index);
    } else {
      newHidden.add(index);
    }
    setHidden(newHidden);
  };

  const visibleItems = items.filter((_, i) => !hidden.has(i));

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h4 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase">
          Sources ({visibleItems.length})
        </h4>
        {hidden.size > 0 && (
          <button
            onClick={() => setHidden(new Set())}
            className="text-xs text-blue-500 hover:underline"
          >
            Show all ({hidden.size} hidden)
          </button>
        )}
      </div>

      {items.map((source, index) => {
        if (hidden.has(index)) return null;

        return (
          <div
            key={index}
            draggable
            onDragStart={() => handleDragStart(index)}
            onDragOver={(e) => handleDragOver(e, index)}
            onDragEnd={handleDragEnd}
            className={`flex items-start gap-2 p-3 rounded-lg border transition cursor-move
              ${pinned.has(index)
                ? "border-blue-300 bg-blue-50 dark:bg-blue-950 dark:border-blue-700"
                : "border-gray-200 bg-white dark:bg-gray-800 dark:border-gray-700"
              }
              ${dragIndex === index ? "opacity-50" : ""}
              hover:shadow-sm
            `}
          >
            {/* Drag handle */}
            <GripVertical className="w-4 h-4 text-gray-300 flex-shrink-0 mt-1" />

            {/* Content */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-medium text-gray-700 dark:text-gray-300">
                  [{source.filename}, p.{source.page}]
                </span>
                <span className="text-xs text-gray-400">
                  {(source.final_score * 100).toFixed(0)}% match
                </span>
              </div>
              <p className="text-xs text-gray-600 dark:text-gray-400 line-clamp-2">
                {source.text}
              </p>
            </div>

            {/* Actions */}
            <div className="flex flex-col gap-1 flex-shrink-0">
              <button
                onClick={() => togglePin(index)}
                className={`p-1 rounded ${
                  pinned.has(index) ? "text-blue-500" : "text-gray-300 hover:text-blue-400"
                }`}
                title="Pin source"
              >
                <Pin className="w-3 h-3" />
              </button>
              <button
                onClick={() => toggleHidden(index)}
                className="p-1 rounded text-gray-300 hover:text-red-400"
                title="Hide source"
              >
                <EyeOff className="w-3 h-3" />
              </button>
              {source.filename.endsWith(".pdf") && onViewPdf && (
                <button
                  onClick={() => onViewPdf(source.filename, source.page)}
                  className="p-1 rounded text-gray-300 hover:text-green-400"
                  title="View in PDF"
                >
                  <ExternalLink className="w-3 h-3" />
                </button>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
