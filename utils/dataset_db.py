"""
SQLite database interface for managing projects and datasets.
Supports project-based organization, dataset merging, and working table management.
"""
import sqlite3
import json
import pandas as pd
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime
import os
import streamlit as st


class DatasetDB:
    """SQLite database interface for project and dataset management."""
    
    def __init__(self, db_path: str = "datasets.db"):
        """Initialize database connection and create tables if needed."""
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Create tables if they don't exist and migrate existing tables.
        
        This method is designed to be robust - if schema migration fails,
        it will recreate the database from scratch.
        """
        try:
            self._migrate_or_create_tables()
        except Exception as e:
            # If migration fails, try to reset and recreate
            import logging
            logging.warning(f"Database migration failed: {e}. Recreating database.")
            self._force_reset_db()
            self._migrate_or_create_tables()
    
    def _migrate_or_create_tables(self):
        """Attempt to create or migrate database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Projects table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    active BOOLEAN NOT NULL DEFAULT 0
                )
            """)
            
            # Check if datasets table exists and needs migration
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='datasets'")
            datasets_exists = cursor.fetchone() is not None
            
            if datasets_exists:
                # Check columns and add missing ones
                cursor.execute("PRAGMA table_info(datasets)")
                columns = [row[1] for row in cursor.fetchall()]
                
                if 'project_id' not in columns:
                    cursor.execute("ALTER TABLE datasets ADD COLUMN project_id INTEGER")
                
                if 'column_types' not in columns:
                    cursor.execute("ALTER TABLE datasets ADD COLUMN column_types TEXT")
                
                if 'is_transposed' not in columns:
                    cursor.execute("ALTER TABLE datasets ADD COLUMN is_transposed BOOLEAN DEFAULT 0")
            else:
                # Create new datasets table with full schema
                cursor.execute("""
                    CREATE TABLE datasets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        project_id INTEGER,
                        name TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        file_type TEXT NOT NULL,
                        shape_rows INTEGER NOT NULL,
                        shape_cols INTEGER NOT NULL,
                        columns TEXT NOT NULL,
                        column_types TEXT,
                        upload_timestamp TEXT NOT NULL,
                        is_transposed BOOLEAN NOT NULL DEFAULT 0,
                        FOREIGN KEY (project_id) REFERENCES projects(id)
                    )
                """)
            
            # Merge configurations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS merge_configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    merge_steps TEXT NOT NULL,
                    result_shape_rows INTEGER,
                    result_shape_cols INTEGER,
                    result_columns TEXT,
                    created_at TEXT NOT NULL,
                    is_working_table BOOLEAN NOT NULL DEFAULT 0,
                    FOREIGN KEY (project_id) REFERENCES projects(id)
                )
            """)
            
            conn.commit()
        finally:
            conn.close()
    
    def _force_reset_db(self):
        """Force reset the database by deleting the file."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
    
    def reset_all_data(self) -> bool:
        """
        Reset the entire database - deletes all projects, datasets, and merge configs.
        Returns True if successful.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM merge_configs")
            cursor.execute("DELETE FROM datasets")
            cursor.execute("DELETE FROM projects")
            
            # Reset auto-increment counters
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='merge_configs'")
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='datasets'")
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='projects'")
            
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the database for display to users."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM projects")
        n_projects = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM datasets")
        n_datasets = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM merge_configs")
        n_merges = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'n_projects': n_projects,
            'n_datasets': n_datasets,
            'n_merge_configs': n_merges,
            'db_path': self.db_path,
            'db_exists': os.path.exists(self.db_path),
            'db_size_kb': round(os.path.getsize(self.db_path) / 1024, 1) if os.path.exists(self.db_path) else 0
        }
    
    # =========================================================================
    # PROJECT MANAGEMENT
    # =========================================================================
    
    def create_project(self, name: str, description: str = "") -> int:
        """Create a new project and return its ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.utcnow().isoformat()
        
        # Deactivate all other projects
        cursor.execute("UPDATE projects SET active = 0")
        
        cursor.execute("""
            INSERT INTO projects (name, description, created_at, updated_at, active)
            VALUES (?, ?, ?, ?, 1)
        """, (name, description, now, now))
        
        project_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return project_id
    
    def get_project(self, project_id: int) -> Optional[Dict[str, Any]]:
        """Get project by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, description, created_at, updated_at, active
            FROM projects WHERE id = ?
        """, (project_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        return {
            'id': row[0],
            'name': row[1],
            'description': row[2],
            'created_at': row[3],
            'updated_at': row[4],
            'active': bool(row[5])
        }
    
    def get_all_projects(self) -> List[Dict[str, Any]]:
        """Get all projects."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, description, created_at, updated_at, active
            FROM projects ORDER BY updated_at DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'id': row[0],
                'name': row[1],
                'description': row[2],
                'created_at': row[3],
                'updated_at': row[4],
                'active': bool(row[5])
            }
            for row in rows
        ]
    
    def get_active_project(self) -> Optional[Dict[str, Any]]:
        """Get the currently active project."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, description, created_at, updated_at, active
            FROM projects WHERE active = 1 LIMIT 1
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        return {
            'id': row[0],
            'name': row[1],
            'description': row[2],
            'created_at': row[3],
            'updated_at': row[4],
            'active': bool(row[5])
        }
    
    def set_active_project(self, project_id: int) -> bool:
        """Set a project as active."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM projects WHERE id = ?", (project_id,))
        if cursor.fetchone() is None:
            conn.close()
            return False
        
        cursor.execute("UPDATE projects SET active = 0")
        cursor.execute("UPDATE projects SET active = 1 WHERE id = ?", (project_id,))
        
        conn.commit()
        conn.close()
        return True
    
    def delete_project(self, project_id: int) -> bool:
        """Delete a project and all its datasets and merge configs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete merge configs
        cursor.execute("DELETE FROM merge_configs WHERE project_id = ?", (project_id,))
        # Delete datasets
        cursor.execute("DELETE FROM datasets WHERE project_id = ?", (project_id,))
        # Delete project
        cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted
    
    # =========================================================================
    # DATASET MANAGEMENT
    # =========================================================================
    
    def add_dataset(
        self,
        project_id: int,
        name: str,
        filename: str,
        file_type: str,
        shape_rows: int,
        shape_cols: int,
        columns: List[str],
        column_types: Optional[Dict[str, str]] = None,
        is_transposed: bool = False
    ) -> int:
        """Add a dataset to a project."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO datasets (
                project_id, name, filename, file_type, shape_rows, shape_cols,
                columns, column_types, upload_timestamp, is_transposed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            project_id,
            name,
            filename,
            file_type,
            shape_rows,
            shape_cols,
            json.dumps(columns),
            json.dumps(column_types) if column_types else None,
            datetime.utcnow().isoformat(),
            1 if is_transposed else 0
        ))
        
        dataset_id = cursor.lastrowid
        
        # Update project timestamp
        cursor.execute(
            "UPDATE projects SET updated_at = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), project_id)
        )
        
        conn.commit()
        conn.close()
        
        return dataset_id
    
    def get_dataset(self, dataset_id: int) -> Optional[Dict[str, Any]]:
        """Get dataset by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, project_id, name, filename, file_type, shape_rows, shape_cols,
                   columns, column_types, upload_timestamp, is_transposed
            FROM datasets WHERE id = ?
        """, (dataset_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        return {
            'id': row[0],
            'project_id': row[1],
            'name': row[2],
            'filename': row[3],
            'file_type': row[4],
            'shape_rows': row[5],
            'shape_cols': row[6],
            'columns': json.loads(row[7]),
            'column_types': json.loads(row[8]) if row[8] else None,
            'upload_timestamp': row[9],
            'is_transposed': bool(row[10])
        }
    
    def get_project_datasets(self, project_id: int) -> List[Dict[str, Any]]:
        """Get all datasets for a project."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, project_id, name, filename, file_type, shape_rows, shape_cols,
                   columns, column_types, upload_timestamp, is_transposed
            FROM datasets WHERE project_id = ?
            ORDER BY upload_timestamp DESC
        """, (project_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'id': row[0],
                'project_id': row[1],
                'name': row[2],
                'filename': row[3],
                'file_type': row[4],
                'shape_rows': row[5],
                'shape_cols': row[6],
                'columns': json.loads(row[7]),
                'column_types': json.loads(row[8]) if row[8] else None,
                'upload_timestamp': row[9],
                'is_transposed': bool(row[10])
            }
            for row in rows
        ]
    
    def delete_dataset(self, dataset_id: int) -> bool:
        """Delete a dataset."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        return deleted
    
    # =========================================================================
    # MERGE CONFIGURATION MANAGEMENT
    # =========================================================================
    
    def save_merge_config(
        self,
        project_id: int,
        name: str,
        merge_steps: List[Dict[str, Any]],
        result_shape: Tuple[int, int],
        result_columns: List[str],
        set_as_working: bool = True
    ) -> int:
        """Save a merge configuration."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # If setting as working table, unset others
        if set_as_working:
            cursor.execute(
                "UPDATE merge_configs SET is_working_table = 0 WHERE project_id = ?",
                (project_id,)
            )
        
        cursor.execute("""
            INSERT INTO merge_configs (
                project_id, name, merge_steps, result_shape_rows, result_shape_cols,
                result_columns, created_at, is_working_table
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            project_id,
            name,
            json.dumps(merge_steps),
            result_shape[0],
            result_shape[1],
            json.dumps(result_columns),
            datetime.utcnow().isoformat(),
            1 if set_as_working else 0
        ))
        
        config_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return config_id
    
    def get_merge_config(self, config_id: int) -> Optional[Dict[str, Any]]:
        """Get merge config by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, project_id, name, merge_steps, result_shape_rows, result_shape_cols,
                   result_columns, created_at, is_working_table
            FROM merge_configs WHERE id = ?
        """, (config_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        return {
            'id': row[0],
            'project_id': row[1],
            'name': row[2],
            'merge_steps': json.loads(row[3]),
            'result_shape_rows': row[4],
            'result_shape_cols': row[5],
            'result_columns': json.loads(row[6]),
            'created_at': row[7],
            'is_working_table': bool(row[8])
        }
    
    def get_project_merge_configs(self, project_id: int) -> List[Dict[str, Any]]:
        """Get all merge configs for a project."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, project_id, name, merge_steps, result_shape_rows, result_shape_cols,
                   result_columns, created_at, is_working_table
            FROM merge_configs WHERE project_id = ?
            ORDER BY created_at DESC
        """, (project_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'id': row[0],
                'project_id': row[1],
                'name': row[2],
                'merge_steps': json.loads(row[3]),
                'result_shape_rows': row[4],
                'result_shape_cols': row[5],
                'result_columns': json.loads(row[6]),
                'created_at': row[7],
                'is_working_table': bool(row[8])
            }
            for row in rows
        ]
    
    def get_working_table_config(self, project_id: int) -> Optional[Dict[str, Any]]:
        """Get the current working table merge config for a project."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, project_id, name, merge_steps, result_shape_rows, result_shape_cols,
                   result_columns, created_at, is_working_table
            FROM merge_configs WHERE project_id = ? AND is_working_table = 1
            LIMIT 1
        """, (project_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        return {
            'id': row[0],
            'project_id': row[1],
            'name': row[2],
            'merge_steps': json.loads(row[3]),
            'result_shape_rows': row[4],
            'result_shape_cols': row[5],
            'result_columns': json.loads(row[6]),
            'created_at': row[7],
            'is_working_table': bool(row[8])
        }
    
    def set_working_table(self, project_id: int, config_id: int) -> bool:
        """Set a merge config as the working table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Unset all
        cursor.execute(
            "UPDATE merge_configs SET is_working_table = 0 WHERE project_id = ?",
            (project_id,)
        )
        # Set the selected one
        cursor.execute(
            "UPDATE merge_configs SET is_working_table = 1 WHERE id = ? AND project_id = ?",
            (config_id, project_id)
        )
        
        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return updated
    
    def delete_merge_config(self, config_id: int) -> bool:
        """Delete a merge config."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM merge_configs WHERE id = ?", (config_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        return deleted


# =========================================================================
# MERGE UTILITIES
# =========================================================================

def detect_common_columns(datasets: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Detect columns that appear in multiple datasets.
    
    Returns:
        Dict mapping column name -> list of dataset names that have this column
    """
    column_datasets = {}
    
    for dataset in datasets:
        dataset_name = dataset['name']
        for col in dataset['columns']:
            # Ensure column name is a string (pandas allows int column names)
            col_str = str(col)
            if col_str not in column_datasets:
                column_datasets[col_str] = []
            column_datasets[col_str].append(dataset_name)
    
    # Filter to columns that appear in more than one dataset
    common = {
        col: ds_list
        for col, ds_list in column_datasets.items()
        if len(ds_list) > 1
    }
    
    return common


def suggest_join_keys(datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Suggest potential join keys between datasets.
    
    Returns:
        List of suggested joins with confidence scores
    """
    suggestions = []
    common_cols = detect_common_columns(datasets)
    
    # Create dataset name -> columns mapping (ensure all column names are strings)
    ds_cols = {d['name']: set(str(c) for c in d['columns']) for d in datasets}
    
    # For each pair of datasets
    dataset_names = list(ds_cols.keys())
    for i, ds1 in enumerate(dataset_names):
        for ds2 in dataset_names[i+1:]:
            # Find shared columns
            shared = ds_cols[ds1] & ds_cols[ds2]
            
            for col in shared:
                # Score based on column name patterns (higher = more likely a key)
                score = 0
                col_lower = str(col).lower()  # Ensure string conversion
                
                # ID-like patterns
                if 'id' in col_lower or '_id' in col_lower:
                    score += 3
                if col_lower.endswith('_key') or col_lower.endswith('key'):
                    score += 2
                if col_lower in ['id', 'key', 'index', 'code']:
                    score += 2
                
                # Common entity identifiers
                if any(x in col_lower for x in ['patient', 'user', 'customer', 'account', 'order', 'product']):
                    score += 1
                
                # Date/time columns often used for joins
                if any(x in col_lower for x in ['date', 'time', 'timestamp']):
                    score += 1
                
                suggestions.append({
                    'left_dataset': ds1,
                    'right_dataset': ds2,
                    'join_column': col,
                    'confidence': min(score, 5) / 5.0  # Normalize to 0-1
                })
    
    # Sort by confidence
    suggestions.sort(key=lambda x: x['confidence'], reverse=True)
    
    return suggestions


def execute_merge(
    dataframes: Dict[str, pd.DataFrame],
    merge_steps: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Execute a series of merge steps.
    
    Args:
        dataframes: Dict mapping dataset name -> DataFrame
        merge_steps: List of merge operations, each with:
            - left: dataset name or 'result' (for chained merges)
            - right: dataset name
            - on: column(s) to join on (string or list)
            - how: 'left', 'right', 'inner', 'outer'
            - suffixes: tuple of suffixes for overlapping columns
    
    Returns:
        Merged DataFrame
    """
    if not merge_steps:
        # Return the first dataframe if no merge steps
        df = next(iter(dataframes.values())).copy()
        df.columns = [str(c) for c in df.columns]
        return df
    
    # Ensure all dataframes have string column names for merge compatibility
    dfs = {}
    for name, df in dataframes.items():
        df_copy = df.copy()
        df_copy.columns = [str(c) for c in df_copy.columns]
        dfs[name] = df_copy
    
    result = None
    
    for step in merge_steps:
        left_name = step['left']
        right_name = step['right']
        on = step.get('on')
        how = step.get('how', 'inner')
        suffixes = step.get('suffixes', ('_x', '_y'))
        
        # Get left dataframe
        if left_name == 'result':
            if result is None:
                raise ValueError("Cannot use 'result' as left table in first merge step")
            left_df = result
        else:
            left_df = dfs[left_name]
        
        # Get right dataframe
        right_df = dfs[right_name]
        
        # Handle different join key specifications
        left_on = step.get('left_on')
        right_on = step.get('right_on')
        
        # Ensure join keys are strings
        if left_on:
            left_on = str(left_on)
        if right_on:
            right_on = str(right_on)
        if on:
            on = str(on) if isinstance(on, (str, int)) else [str(c) for c in on]
        
        if left_on and right_on:
            result = pd.merge(
                left_df, right_df,
                left_on=left_on, right_on=right_on,
                how=how, suffixes=suffixes
            )
        elif on:
            result = pd.merge(
                left_df, right_df,
                on=on, how=how, suffixes=suffixes
            )
        else:
            raise ValueError("Must specify 'on' or both 'left_on' and 'right_on'")
    
    return result


# =========================================================================
# GLOBAL INSTANCE
# =========================================================================

_db_instance: Optional[DatasetDB] = None


def get_db() -> DatasetDB:
    """Get the global database instance.
    
    Uses Streamlit session state to ensure proper initialization per session.
    """
    global _db_instance
    
    # Check if we need to reinitialize (e.g., after code changes)
    if hasattr(st, 'session_state'):
        db_path = st.session_state.get('dataset_db_path', 'datasets.db')
        
        # Force new instance if path changed or first time in this session
        if '_db_initialized' not in st.session_state:
            _db_instance = None
            st.session_state._db_initialized = True
    else:
        db_path = 'datasets.db'
    
    if _db_instance is None:
        _db_instance = DatasetDB(db_path)
    
    return _db_instance


def reset_db():
    """Reset the global database instance (forces re-initialization)."""
    global _db_instance
    _db_instance = None
    if hasattr(st, 'session_state') and '_db_initialized' in st.session_state:
        del st.session_state._db_initialized
