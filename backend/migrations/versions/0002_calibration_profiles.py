"""calibration profiles

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-03 01:34:13.209555
"""
import sqlalchemy as sa
from alembic import op

revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('calibration_profiles',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('model_version', sa.String(length=12), nullable=False),
    sa.Column('labels', sa.JSON(), nullable=False),
    sa.Column('prototypes', sa.JSON(), nullable=False),
    sa.Column('samples_per_label', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id')
    )


def downgrade() -> None:
    op.drop_table('calibration_profiles')
