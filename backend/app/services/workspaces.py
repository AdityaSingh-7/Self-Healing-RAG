"""
services/workspaces.py — Shared Workspaces

Allows multiple users to share a document collection.
Uses SQLite for workspace management (lightweight, no extra infra).

CONCEPTS:
- Workspace: A shared document collection with a name and members
- Owner: The user who created the workspace (can invite/remove members)
- Member: A user who can upload docs to and query a workspace
- Viewer: A user who can query but not upload

Pinecone namespace strategy:
  Personal docs: namespace = "user_{user_id}"
  Workspace docs: namespace = "ws_{workspace_id}"
"""

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "analytics.db"


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_workspaces_db():
    """Create workspace tables if they don't exist."""
    conn = _get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS workspaces (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            owner_id TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS workspace_members (
            workspace_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'member',
            joined_at TEXT NOT NULL,
            PRIMARY KEY (workspace_id, user_id),
            FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS workspace_documents (
            workspace_id TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            uploaded_by TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            summary TEXT DEFAULT '',
            pages INTEGER DEFAULT 0,
            chunks INTEGER DEFAULT 0,
            PRIMARY KEY (workspace_id, doc_id),
            FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
        );
    """)
    conn.commit()
    conn.close()


def create_workspace(name: str, description: str, owner_id: str) -> dict:
    """Create a new workspace. Owner is automatically added as admin."""
    workspace_id = str(uuid.uuid4())[:8]  # Short ID for readability
    now = datetime.now(timezone.utc).isoformat()

    conn = _get_connection()
    conn.execute(
        "INSERT INTO workspaces (id, name, description, owner_id, created_at) VALUES (?, ?, ?, ?, ?)",
        (workspace_id, name, description, owner_id, now),
    )
    conn.execute(
        "INSERT INTO workspace_members (workspace_id, user_id, role, joined_at) VALUES (?, ?, 'admin', ?)",
        (workspace_id, owner_id, now),
    )
    conn.commit()
    conn.close()

    return {
        "id": workspace_id,
        "name": name,
        "description": description,
        "owner_id": owner_id,
        "created_at": now,
        "namespace": f"ws_{workspace_id}",
    }


def add_member(workspace_id: str, user_id: str, role: str = "member", requester_id: str = "") -> bool:
    """Add a member to a workspace. Only admins/owners can add."""
    conn = _get_connection()

    # Check if requester is admin/owner
    row = conn.execute(
        "SELECT role FROM workspace_members WHERE workspace_id = ? AND user_id = ?",
        (workspace_id, requester_id),
    ).fetchone()

    if not row or row["role"] not in ("admin", "owner"):
        conn.close()
        return False

    now = datetime.now(timezone.utc).isoformat()
    try:
        conn.execute(
            "INSERT INTO workspace_members (workspace_id, user_id, role, joined_at) VALUES (?, ?, ?, ?)",
            (workspace_id, user_id, role, now),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return False  # Already a member

    conn.close()
    return True


def remove_member(workspace_id: str, user_id: str, requester_id: str) -> bool:
    """Remove a member from a workspace."""
    conn = _get_connection()

    # Check requester is admin
    row = conn.execute(
        "SELECT role FROM workspace_members WHERE workspace_id = ? AND user_id = ?",
        (workspace_id, requester_id),
    ).fetchone()

    if not row or row["role"] != "admin":
        conn.close()
        return False

    conn.execute(
        "DELETE FROM workspace_members WHERE workspace_id = ? AND user_id = ?",
        (workspace_id, user_id),
    )
    conn.commit()
    conn.close()
    return True


def get_user_workspaces(user_id: str) -> list[dict]:
    """Get all workspaces a user belongs to."""
    conn = _get_connection()
    rows = conn.execute("""
        SELECT w.*, wm.role,
            (SELECT COUNT(*) FROM workspace_members WHERE workspace_id = w.id) as member_count,
            (SELECT COUNT(*) FROM workspace_documents WHERE workspace_id = w.id) as doc_count
        FROM workspaces w
        JOIN workspace_members wm ON w.id = wm.workspace_id
        WHERE wm.user_id = ?
        ORDER BY w.created_at DESC
    """, (user_id,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_workspace_members(workspace_id: str) -> list[dict]:
    """Get all members of a workspace."""
    conn = _get_connection()
    rows = conn.execute(
        "SELECT user_id, role, joined_at FROM workspace_members WHERE workspace_id = ?",
        (workspace_id,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def is_member(workspace_id: str, user_id: str) -> bool:
    """Check if a user is a member of a workspace."""
    conn = _get_connection()
    row = conn.execute(
        "SELECT 1 FROM workspace_members WHERE workspace_id = ? AND user_id = ?",
        (workspace_id, user_id),
    ).fetchone()
    conn.close()
    return row is not None


def get_workspace_namespace(workspace_id: str) -> str:
    """Get the Pinecone namespace for a workspace."""
    return f"ws_{workspace_id}"


def add_document_to_workspace(workspace_id: str, doc_id: str, filename: str, uploaded_by: str, summary: str = "", pages: int = 0, chunks: int = 0):
    """Record a document upload in a workspace."""
    conn = _get_connection()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO workspace_documents (workspace_id, doc_id, filename, uploaded_by, uploaded_at, summary, pages, chunks) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (workspace_id, doc_id, filename, uploaded_by, now, summary, pages, chunks),
    )
    conn.commit()
    conn.close()


def get_workspace_documents(workspace_id: str) -> list[dict]:
    """Get all documents in a workspace."""
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM workspace_documents WHERE workspace_id = ? ORDER BY uploaded_at DESC",
        (workspace_id,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def delete_workspace(workspace_id: str, requester_id: str) -> bool:
    """Delete a workspace. Only the owner can delete."""
    conn = _get_connection()
    row = conn.execute(
        "SELECT owner_id FROM workspaces WHERE id = ?", (workspace_id,)
    ).fetchone()

    if not row or row["owner_id"] != requester_id:
        conn.close()
        return False

    conn.execute("DELETE FROM workspaces WHERE id = ?", (workspace_id,))
    conn.commit()
    conn.close()
    return True


# Initialize on import
init_workspaces_db()
