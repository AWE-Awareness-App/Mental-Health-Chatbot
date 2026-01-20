"""
Database Migration Script: Fix Message Counts
==============================================

Problem: Users' total_messages field was not incremented on first interaction,
causing incorrect counts in the dashboard.

This script:
1. Counts actual messages per user in the Message table
2. Updates each user's total_messages field with the correct count

Run this ONCE to fix existing data.
"""

import os
import logging
from sqlalchemy import text, func
from sqlalchemy.orm import sessionmaker
from database_aad import get_database_engine
from database import aichatusers, Message

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fix_message_counts():
    """Fix total_messages count for all existing users."""

    engine = get_database_engine()
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = SessionLocal()

    try:
        logger.info("Starting message count fix...")

        # Get all users
        users = db.query(aichatusers).all()
        logger.info(f"Found {len(users)} users to process")

        updated_count = 0

        for user in users:
            # Count actual messages for this user in Message table
            actual_message_count = db.query(Message).filter(
                Message.user_id == user.id,
                Message.role == 'user'  # Only count user messages, not assistant responses
            ).count()

            # Update if different
            if user.total_messages != actual_message_count:
                old_count = user.total_messages
                user.total_messages = actual_message_count
                db.commit()
                updated_count += 1
                logger.info(
                    f"Updated user {user.whatsapp_number[:20]}... "
                    f"from {old_count} to {actual_message_count} messages"
                )

        logger.info(f"\n✅ Migration complete!")
        logger.info(f"   Total users: {len(users)}")
        logger.info(f"   Updated: {updated_count}")
        logger.info(f"   Already correct: {len(users) - updated_count}")

        # Show summary
        total_messages = db.query(func.sum(aichatusers.total_messages)).scalar() or 0
        logger.info(f"\n📊 New totals:")
        logger.info(f"   Total messages across all users: {total_messages}")

    except Exception as e:
        logger.error(f"❌ Error during migration: {e}", exc_info=True)
        db.rollback()
        raise

    finally:
        db.close()
        engine.dispose()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔧 MESSAGE COUNT FIX MIGRATION")
    print("="*60)
    print("\nThis will update total_messages for all users based on")
    print("actual message counts in the database.")
    print("\n⚠️  Make sure to backup your database before running!")
    print("="*60 + "\n")

    response = input("Continue? (yes/no): ").strip().lower()

    if response == 'yes':
        fix_message_counts()
    else:
        print("❌ Migration cancelled.")
