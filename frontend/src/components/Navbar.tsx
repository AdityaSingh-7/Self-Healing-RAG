"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useSession, signIn, signOut } from "next-auth/react";
import { MessageSquare, Upload, LogIn, LogOut, FileText } from "lucide-react";

export function Navbar() {
  const pathname = usePathname();
  const { data: session } = useSession();

  const links = [
    { href: "/chat", label: "Chat", icon: MessageSquare },
    { href: "/upload", label: "Upload", icon: Upload },
  ];

  return (
    <nav className="bg-white border-b px-6 py-3 flex items-center justify-between">
      {/* Logo */}
      <Link href="/" className="flex items-center gap-2">
        <FileText className="w-6 h-6 text-blue-600" />
        <span className="font-bold text-lg">RAG System</span>
      </Link>

      {/* Navigation Links */}
      <div className="flex items-center gap-1">
        {links.map(({ href, label, icon: Icon }) => (
          <Link
            key={href}
            href={href}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition ${
              pathname === href
                ? "bg-blue-50 text-blue-700"
                : "text-gray-600 hover:bg-gray-100"
            }`}
          >
            <Icon className="w-4 h-4" />
            {label}
          </Link>
        ))}
      </div>

      {/* Auth */}
      <div className="flex items-center gap-3">
        {session ? (
          <>
            <span className="text-sm text-gray-600">
              {session.user?.name || session.user?.email}
            </span>
            <button
              onClick={() => signOut()}
              className="flex items-center gap-1 px-3 py-2 text-sm text-gray-600 hover:bg-gray-100 rounded-lg transition"
            >
              <LogOut className="w-4 h-4" />
              Sign Out
            </button>
          </>
        ) : (
          <button
            onClick={() => signIn()}
            className="flex items-center gap-1 px-4 py-2 text-sm bg-gray-900 text-white rounded-lg hover:bg-gray-800 transition"
          >
            <LogIn className="w-4 h-4" />
            Sign In
          </button>
        )}
      </div>
    </nav>
  );
}
