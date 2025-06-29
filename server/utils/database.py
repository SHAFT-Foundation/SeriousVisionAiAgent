"""
Database configuration and connection management for Vision Agent
"""
import os
import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool
from contextlib import asynccontextmanager

from ..models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and sessions"""
    
    def __init__(self, database_url: str = None, echo: bool = False):
        self.database_url = database_url or os.getenv(
            "DATABASE_URL", 
            "postgresql+asyncpg://user:password@localhost:5432/vision_agent"
        )
        
        # Create async engine
        self.engine = create_async_engine(
            self.database_url,
            echo=echo,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
            poolclass=NullPool if "sqlite" in self.database_url else None
        )
        
        # Create session factory
        self.async_session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def create_tables(self) -> None:
        """Create all database tables"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    async def drop_tables(self) -> None:
        """Drop all database tables (use with caution!)"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise
    
    async def close(self) -> None:
        """Close database connections"""
        await self.engine.dispose()
        logger.info("Database connections closed")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with automatic cleanup"""
        async with self.async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def get_session_dependency(self) -> AsyncGenerator[AsyncSession, None]:
        """FastAPI dependency for database sessions"""
        async with self.get_session() as session:
            yield session


# Global database manager instance
db_manager = DatabaseManager()


# FastAPI dependency function
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency to get database session"""
    async with db_manager.get_session() as session:
        yield session


# Utility functions for common database operations
async def init_database() -> None:
    """Initialize database with tables"""
    await db_manager.create_tables()


async def close_database() -> None:
    """Close database connections"""
    await db_manager.close()