"""
Utilities for RSA encryption and decryption of media files.
"""

import os
import base64
from typing import Tuple, Optional, Union
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization


def generate_rsa_key_pair(key_size: int = 2048) -> Tuple[bytes, bytes]:
    """
    Generate an RSA key pair and return the private and public keys.
    
    Args:
        key_size: Size of the RSA key in bits (default: 2048)
        
    Returns:
        Tuple of (private_key_pem, public_key_pem)
    """
    # Generate a private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size,
    )

    # Extract the public key
    public_key = private_key.public_key()

    # Serialize private key to PEM format
    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

    # Serialize public key to PEM format
    public_key_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

    return private_key_pem, public_key_pem


def load_key(key_path: str) -> Union[
    serialization.RSAPublicKey, 
    serialization.RSAPrivateKey
]:
    """
    Load a key from a PEM file.
    
    Args:
        key_path: Path to the PEM file
        
    Returns:
        An RSA public or private key object
    """
    with open(key_path, "rb") as key_file:
        key_data = key_file.read()
        
        try:
            # Try loading as a private key first
            return serialization.load_pem_private_key(
                key_data,
                password=None,
            )
        except ValueError:
            # If that fails, try loading as a public key
            return serialization.load_pem_public_key(
                key_data
            )


def encrypt_data(data: bytes, public_key_path: str) -> str:
    """
    Encrypt binary data using RSA encryption with a public key.
    
    Args:
        data: Binary data to encrypt
        public_key_path: Path to the public key PEM file
        
    Returns:
        Base64-encoded encrypted data
    """
    # For large files like images or videos, we need to encrypt in chunks
    # since RSA can only encrypt data smaller than the key size
    
    # Load the public key
    public_key = load_key(public_key_path)
    
    # Maximum size that can be encrypted with RSA (key_size / 8 - padding)
    # For a 2048-bit key, this is roughly 190 bytes
    chunk_size = 190
    
    encrypted_chunks = []
    
    # Process data in chunks
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        
        # Encrypt the chunk
        encrypted_chunk = public_key.encrypt(
            chunk,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        encrypted_chunks.append(encrypted_chunk)
    
    # Join encrypted chunks and base64 encode
    combined_data = b''.join(encrypted_chunks)
    return base64.b64encode(combined_data).decode('utf-8')


def decrypt_data(encrypted_data: str, private_key_path: str) -> bytes:
    """
    Decrypt base64-encoded RSA-encrypted data using a private key.
    
    Args:
        encrypted_data: Base64-encoded encrypted data
        private_key_path: Path to the private key PEM file
        
    Returns:
        Decrypted binary data
    """
    # Load the private key
    private_key = load_key(private_key_path)
    
    # Base64 decode
    encrypted_bytes = base64.b64decode(encrypted_data)
    
    # Get the size of the key in bytes
    key_size_bytes = private_key.key_size // 8
    
    # Process data in chunks
    decrypted_chunks = []
    
    # Process data in chunks of key_size
    for i in range(0, len(encrypted_bytes), key_size_bytes):
        chunk = encrypted_bytes[i:i + key_size_bytes]
        
        if len(chunk) == key_size_bytes:  # Make sure we have a full chunk
            # Decrypt the chunk
            decrypted_chunk = private_key.decrypt(
                chunk,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            decrypted_chunks.append(decrypted_chunk)
    
    # Join decrypted chunks
    return b''.join(decrypted_chunks)


def ensure_keys_exist(keys_dir: str = "keys") -> None:
    """
    Ensure that RSA keys exist, generating them if needed.
    
    Args:
        keys_dir: Directory to store the keys
    """
    os.makedirs(keys_dir, exist_ok=True)
    
    server_private_key_path = os.path.join(keys_dir, "server_private_key.pem")
    client_public_key_path = os.path.join(keys_dir, "client_public_key.pem")
    
    # Check if server private key exists, generate if not
    if not os.path.exists(server_private_key_path):
        print("Generating server key pair...")
        private_key, public_key = generate_rsa_key_pair()
        
        with open(server_private_key_path, "wb") as f:
            f.write(private_key)
            
        # This would be shared with the client in a real application
        with open(os.path.join(keys_dir, "server_public_key.pem"), "wb") as f:
            f.write(public_key)
            
    # For demo purposes, generate client keys if they don't exist
    if not os.path.exists(client_public_key_path):
        print("Generating client key pair (for demo purposes)...")
        private_key, public_key = generate_rsa_key_pair()
        
        with open(client_public_key_path, "wb") as f:
            f.write(public_key)
            
        # This would stay with the client in a real application
        with open(os.path.join(keys_dir, "client_private_key.pem"), "wb") as f:
            f.write(private_key)

    print("Keys ready for encryption/decryption.")
