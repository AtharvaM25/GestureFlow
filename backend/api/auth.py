from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db import get_db
from backend.deps import current_user, login_rate_limit
from backend.models import User
from backend.schemas import Credentials, PasswordConfirm, TokenOut, UserOut
from backend.security import create_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED,
             dependencies=[Depends(login_rate_limit)])
def register(body: Credentials, db: Session = Depends(get_db)):
    email = body.email.lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(status.HTTP_409_CONFLICT, "email already registered")
    user = User(email=email, password_hash=hash_password(body.password))
    db.add(user)
    db.commit()
    return user


@router.post("/login", response_model=TokenOut, dependencies=[Depends(login_rate_limit)])
def login(body: Credentials, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == body.email.lower()))
    # same message whether the email or the password is wrong
    if not verify_password(body.password, user.password_hash if user else None):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "incorrect email or password")
    token, ttl = create_token(user.id)
    return TokenOut(access_token=token, expires_in=ttl)


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)):
    return user


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT,
               dependencies=[Depends(login_rate_limit)])
def delete_account(body: PasswordConfirm, db: Session = Depends(get_db),
                   user: User = Depends(current_user)):
    """Permanently delete the account and everything stored for it: sessions, letters and
    calibration (removed by the foreign keys' ON DELETE CASCADE). Asks for the password
    again so a stolen token alone can't do it."""
    if not verify_password(body.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "incorrect password")
    db.delete(user)
    db.commit()
