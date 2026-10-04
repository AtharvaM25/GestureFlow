"""users, sessions, committed letters

Revision ID: 0001
Revises: 
Create Date: 2026-10-02 23:38:12.011239
"""
import sqlalchemy as sa
from alembic import op

revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('users',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('email', sa.String(length=320), nullable=False),
    sa.Column('password_hash', sa.String(length=255), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('email')
    )
    op.create_table('recognition_sessions',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('model_version', sa.String(length=12), nullable=False),
    sa.Column('text', sa.Text(), server_default='', nullable=False),
    sa.Column('sentence', sa.Text(), nullable=True),
    sa.Column('frame_count', sa.Integer(), server_default='0', nullable=False),
    sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    sa.Column('ended_at', sa.DateTime(timezone=True), nullable=True),
    sa.CheckConstraint('frame_count >= 0', name='ck_sessions_frames'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_sessions_user_started', 'recognition_sessions', ['user_id', 'started_at'], unique=False)
    op.create_table('committed_letters',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('session_id', sa.Integer(), nullable=False),
    sa.Column('gesture', sa.String(length=32), nullable=False),
    sa.Column('confidence', sa.Float(), nullable=False),
    sa.Column('latency_ms', sa.Float(), nullable=False),
    sa.Column('alternatives', sa.JSON(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    sa.CheckConstraint('confidence >= 0 AND confidence <= 1', name='ck_letters_confidence'),
    sa.ForeignKeyConstraint(['session_id'], ['recognition_sessions.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_committed_letters_session_id'), 'committed_letters', ['session_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_committed_letters_session_id'), table_name='committed_letters')
    op.drop_table('committed_letters')
    op.drop_index('ix_sessions_user_started', table_name='recognition_sessions')
    op.drop_table('recognition_sessions')
    op.drop_table('users')
