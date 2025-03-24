"""
Time-Release Blockchain Implementation

This code provides a simplified implementation of a time-release blockchain
with functionality for sending time-locked messages.

Key concepts:
- Time-release cryptography: Messages are encrypted and can only be decrypted after a specified time
- Blockchain: Decentralized ledger maintaining transaction history
- Public/private key pairs: Used for encryption and decryption of messages
"""
import hashlib
import time
import json
import random
import ecdsa
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict

# Security parameter (small for demonstration)
# P = 2**256 - 2**32 - 977  # A prime number used for modular arithmetic in cryptographic operations
P = 2**16 - 2**8 - 55  # A prime number used for modular arithmetic in cryptographic operations

class Transaction:
    def __init__(self, sender: str, recipient: str, message: str, public_key: Tuple[int, int, int]):
        """
        Create a transaction with an encrypted message.
        
        Args:
            sender: Sender's identifier
            recipient: Recipient's identifier
            message: Message to encrypt
            public_key: Public key to encrypt the message with (x, y, z components)
        """
        self.sender = sender
        self.recipient = recipient
        self.timestamp = time.time()  # Record current time as transaction timestamp
        # Encrypt the message with the provided public key
        self.encrypted_message = self.encrypt_message(message, public_key)
        # Generate a unique transaction ID based on transaction details
        self.transaction_id = self.calculate_hash()
    
    def encrypt_message(self, message: str, public_key: Tuple[int, int, int]) -> str:
        """
        Encrypt a message using the provided public key.
        
        This is a simplified encryption method for demonstration purposes.
        In a real implementation, you would use proper encryption algorithms.
        
        Args:
            message: Plain text message to encrypt
            public_key: A tuple of (x, y, z) values used for encryption
            
        Returns:
            Hexadecimal string representation of the encrypted message
        """
        # Convert message to bytes for encryption
        message_bytes = message.encode('utf-8')
        
        # Unpack the public key components
        x, y, z = public_key
        
        # Create a simple encryption by transforming each byte
        encrypted_bytes = bytearray()
        for i, byte in enumerate(message_bytes):
            # Mix the byte with public key components
            # The formula creates a unique transformation for each byte position
            encrypted_byte = (byte + x + (y * i) + z) % 256  # Modulo 256 to keep in byte range
            encrypted_bytes.append(encrypted_byte)
        # Return as a hex string for easy storage and transmission
        return encrypted_bytes.hex()
    
    @staticmethod
    def decrypt_message(encrypted_hex: str, private_key: int) -> str:
        """
        Decrypt a message using the provided private key.
        
        This is a simplified decryption method for demonstration purposes.
        In a real implementation, you would use proper decryption algorithms.
        
        Args:
            encrypted_hex: Hexadecimal string of the encrypted message
            private_key: Private key integer used for decryption
            
        Returns:
            Decrypted message as a string
        """
        # Convert hex string to bytes for decryption
        encrypted_bytes = bytes.fromhex(encrypted_hex)
        
        # Derive decryption parameters from private key
        # Extract components from different parts of the private key
        x = private_key % 10000  # Last 4 digits
        y = (private_key // 10000) % 10000  # Next 4 digits
        z = (private_key // 100000000) % 10000  # Next 4 digits
        
        # Decrypt each byte by reversing the encryption operation
        decrypted_bytes = bytearray()
        for i, byte in enumerate(encrypted_bytes):
            # Reverse the encryption operation from encrypt_message
            decrypted_byte = (byte - x - (y * i) - z) % 256
            decrypted_bytes.append(decrypted_byte)
        
        # Convert back to string and return
        return decrypted_bytes.decode('utf-8')
    
    def calculate_hash(self) -> str:
        """
        Calculate a unique hash for this transaction.
        
        Returns:
            SHA-256 hash of the transaction details as a hexadecimal string
        """
        # Combine all transaction fields into a single string
        transaction_string = f"{self.sender}{self.recipient}{self.timestamp}{self.encrypted_message}"
        # Return SHA-256 hash of the combined string
        return hashlib.sha256(transaction_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        """
        Convert transaction to dictionary for serialization.
        
        Returns:
            Dictionary representation of the transaction
        """
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "timestamp": self.timestamp,
            "encrypted_message": self.encrypted_message,
            "transaction_id": self.transaction_id
        }

@dataclass
class BlockHeader:
    """
    Block header for the time-release blockchain.
    
    Contains metadata about the block including its position in the chain,
    references to previous blocks, and cryptographic elements.
    """
    index: int  # Position of this block in the blockchain
    prev_hash: str  # Hash of the previous block
    timestamp: float  # Time when the block was created
    merkle_root: str  # Root hash of the Merkle tree of transactions
    nonce: int = 0  # Value that will be adjusted during mining to find a valid hash
    public_key: Tuple[int, int, int] = field(default_factory=lambda: (0, 0, 0))  # Public key for this block
    public_key_length: int = 256  # Bit length of the keys
    
    def to_dict(self) -> Dict:
        """
        Convert block header to dictionary for serialization.
        
        Returns:
            Dictionary representation of the block header
        """
        return {
            "index": self.index,
            "prev_hash": self.prev_hash,
            "timestamp": self.timestamp,
            "merkle_root": self.merkle_root,
            "nonce": self.nonce,
            "public_key": list(self.public_key),  # Convert tuple to list for JSON serialization
            "public_key_length": self.public_key_length
        }

class Block:
    def __init__(self, index: int, prev_hash: str, transactions: List[Transaction]):
        """
        Create a new block in the blockchain.
        
        Args:
            index: Block index (position in the chain)
            prev_hash: Hash of the previous block
            transactions: List of transactions to include in this block
        """
        self.transactions = transactions
        # Calculate the Merkle root of the transactions (summary hash)
        self.merkle_root = self.calculate_merkle_root()
        self.timestamp = time.time()  # Record creation time
        
        # Initialize block header with metadata
        self.header = BlockHeader(
            index=index,
            prev_hash=prev_hash,
            timestamp=self.timestamp,
            merkle_root=self.merkle_root
        )
        
        # Block hash will be calculated during mining
        self.hash = ""
    
    def calculate_merkle_root(self) -> str:
        """
        Calculate the Merkle root of transactions.
        
        Merkle tree is a binary tree where each non-leaf node is the hash of its children.
        The root hash summarizes all transactions in the block.
        
        Returns:
            Hexadecimal string representing the Merkle root
        """
        if not self.transactions:
            # If no transactions, return hash of empty string
            return hashlib.sha256("".encode()).hexdigest()
        
        # Get transaction hashes
        tx_hashes = [tx.transaction_id for tx in self.transactions]
        
        # Build the Merkle tree bottom-up
        while len(tx_hashes) > 1:
            # If odd number of hashes, duplicate the last one
            if len(tx_hashes) % 2 != 0:
                tx_hashes.append(tx_hashes[-1])
            
            # Create new level of the tree by hashing pairs of nodes
            temp_hashes = []
            for i in range(0, len(tx_hashes), 2):
                combined = tx_hashes[i] + tx_hashes[i+1]
                temp_hashes.append(hashlib.sha256(combined.encode()).hexdigest())
            
            # Move up to the next level of the tree
            tx_hashes = temp_hashes
        
        # Return the root hash (last remaining hash)
        return tx_hashes[0]
    
    def calculate_hash(self) -> str:
        """
        Calculate hash of the block header.
        
        Uses double SHA-256 hashing (hash of hash) similar to Bitcoin.
        
        Returns:
            Hexadecimal string representing the block hash
        """
        # Serialize the header to a JSON string
        header_string = json.dumps(self.header.to_dict(), sort_keys=True)
        # Apply double SHA-256 hashing
        return hashlib.sha256(hashlib.sha256(header_string.encode()).digest()).hexdigest()
    
    def to_dict(self) -> Dict:
        """
        Convert block to dictionary for serialization.
        
        Returns:
            Dictionary representation of the block
        """
        return {
            "header": self.header.to_dict(),
            "hash": self.hash,
            "transactions": [tx.to_dict() for tx in self.transactions]
        }

class TimeReleaseBlockchain:
    def __init__(self):
        """
        Initialize the blockchain with a genesis block.
        
        The genesis block is the first block in the chain and has special properties.
        """
        self.chain = []  # List to store all blocks
        self.pending_transactions = []  # Transactions waiting to be included in a block
        # self.difficulty = 4  # Number of leading zeros required in block hash (mining difficulty)
        
        # Create the genesis block
        self.create_genesis_block()
    
    def create_genesis_block(self):
        """
        Create the genesis block with an initial public key.
        
        The genesis block is the first block in the blockchain and has special properties:
        - It has no previous block (prev_hash is zeros)
        - It contains no transactions (empty list)
        - It establishes the initial public key for the chain
        """
        # Generate initial public key components
        x = random.randint(10000, 99999)
        y = random.randint(10000, 99999)
        z = random.randint(10000, 99999)
        initial_public_key = (x, y, z)
        
        # Create genesis block with index 0 and empty previous hash
        genesis_block = Block(0, "0" * 64, [])
        genesis_block.header.public_key = initial_public_key
        genesis_block.hash = genesis_block.calculate_hash()
        
        # Add to chain
        self.chain.append(genesis_block)
        print(f"Genesis block created with public key: {initial_public_key}")
    
    def generate_next_public_key(self, previous_public_key: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Generate next public key using the previous public key as a seed.
        
        This implements a deterministic sequence of public keys where each key
        is derived from the previous one, allowing future keys to be predicted.
        
        Args:
            previous_public_key: Tuple of (x, y, z) from the previous block
        
        Returns:
            New public key as a tuple (x, y, z)
        """
        x, y, z = previous_public_key
        
        # Use the previous public key as a seed for the random number generator
        # This ensures deterministic but unpredictable sequence
        random.seed(x * y * z)
        
        # Generate new public key components
        new_x = random.randint(10000, 99999)
        new_y = random.randint(10000, 99999)
        new_z = random.randint(10000, 99999)
        
        return (new_x, new_y, new_z)
    
    def calculate_future_public_keys(self, blocks_ahead: int) -> List[Tuple[int, int, int]]:
        """
        Calculate public keys for future blocks.
        
        This allows messages to be encrypted for future blocks even before those blocks exist.
        
        Args:
            blocks_ahead: Number of future blocks to calculate keys for
            
        Returns:
            List of public keys for future blocks
        """
        if not self.chain:
            return []
        
        future_public_keys = []
        # Start with the public key from the last block in the chain
        current_public_key = self.chain[-1].header.public_key
        
        # Calculate each future public key in sequence
        for _ in range(blocks_ahead):
            next_public_key = self.generate_next_public_key(current_public_key)
            future_public_keys.append(next_public_key)
            current_public_key = next_public_key
        
        return future_public_keys
    
    def add_transaction(self, sender: str, recipient: str, message: str, blocks_ahead: int = 1):
        """
        Add a new transaction to the pending transactions list.
        
        For time-release encryption, the message is encrypted with a future public key
        that will be revealed after the specified number of blocks.
        
        Args:
            sender: Sender's identifier
            recipient: Recipient's identifier
            message: Message to encrypt
            blocks_ahead: Number of blocks to wait before the message can be decrypted
            
        Returns:
            Transaction ID of the created transaction
        """
        # Calculate future public keys based on the number of blocks ahead
        future_public_keys = self.calculate_future_public_keys(blocks_ahead)
        
        if not future_public_keys:
            raise ValueError("Failed to calculate future public keys")
        
        # Create encrypted message with nested encryption
        # Each layer of encryption corresponds to a future block
        encrypted_message = message
        for public_key in reversed(future_public_keys):
            # Apply each layer of encryption, starting from the farthest future block
            tx = Transaction(sender, recipient, encrypted_message, public_key)
            encrypted_message = tx.encrypted_message
        
        # Add the final transaction to pending transactions
        final_tx = Transaction(sender, recipient, encrypted_message, future_public_keys[-1])
        self.pending_transactions.append(final_tx)
        
        print(f"Transaction added: Message will be decryptable after {blocks_ahead} blocks")
        return final_tx.transaction_id
    
    def mine_block(self) -> Block:
        """
        Mine a new block with the pending transactions.
    
        The mining process involves finding a nonce value that results in a block hash
        that satisfies the private key equation:
        SHA256(SHA256(block_header)) ≡ keyprivate (mod p)
    
        In the context of time-release cryptography, the hash of this block will be used
        as the private key for the previous block, allowing decryption of its messages.
    
        Returns:
            The newly mined block
        """
        if not self.chain:
            raise ValueError("Cannot mine block: blockchain is empty")
    
        # Get the last block and its public key
        last_block = self.chain[-1]
        prev_public_key = last_block.header.public_key
    
        # Create a new block with current pending transactions
        new_block = Block(
            index=len(self.chain),
            prev_hash=last_block.hash,
            transactions=self.pending_transactions.copy()
        )
    
        # Generate public key for this block based on the previous block's key
        new_public_key = self.generate_next_public_key(prev_public_key)
        new_block.header.public_key = new_public_key
    
        # For time-release encryption, we need to find a nonce that satisfies:
        # SHA256(SHA256(block_header)) ≡ keyprivate (mod p)
        print("Mining block...")
        start_time = time.time()
    
        # Target private key for this block (derived from previous block's public key)
        # In a real implementation, this would be calculated differently
        x, y, z = prev_public_key
        target_private_key = (x * y * z) % P # Simple calculation for demonstration
    
        # Mining process: find a nonce that produces a hash satisfying the equation
        while True:
            # Try a new nonce
            new_block.header.nonce += 1
            
            # Calculate header string
            header_string = json.dumps(new_block.header.to_dict(), sort_keys=True)
            
            # Calculate double SHA-256 hash
            hash_bytes = hashlib.sha256(hashlib.sha256(header_string.encode()).digest()).digest()
            hash_int = int.from_bytes(hash_bytes, byteorder='big')
            
            # Check if hash satisfies the private key equation: hash ≡ keyprivate (mod p)
            if hash_int % P == target_private_key:
                # Found a valid nonce that satisfies the equation
                block_hash = hash_bytes.hex()
                # private_key = hash_int % P
                private_key = hash_int
                break
    
        mining_time = time.time() - start_time
        print(f"Block mined in {mining_time:.2f} seconds with nonce: {new_block.header.nonce}")
        print(f"Private key for previous block: {private_key}")
        # print(f"Hash satisfies: {hash_int % P} ≡ {target_private_key} (mod {P})")
    
        # Finalize the block with its hash
        new_block.hash = block_hash
    
        # Add to chain and clear pending transactions
        self.chain.append(new_block)
        self.pending_transactions = []
    
        return new_block
    
    def decrypt_message(self, block_index: int, transaction_id: str) -> Optional[str]:
        """
        Attempt to decrypt a message from a specific block and transaction.
        
        The decryption can only succeed if the next block (containing the private key)
        has already been mined.
        
        Args:
            block_index: Index of the block containing the transaction
            transaction_id: ID of the transaction to decrypt
        
        Returns:
            Decrypted message if possible, an error message if not possible
        """
        if block_index >= len(self.chain):
            return None
        
        # Get the block containing the transaction
        block = self.chain[block_index]
        
        # Find the transaction in the block
        transaction = None
        for tx in block.transactions:
            if tx.transaction_id == transaction_id:
                transaction = tx
                break
        
        if not transaction:
            return None
        
        # To decrypt, we need the private key from the next block
        # (which is derived from the next block's hash)
        if block_index + 1 >= len(self.chain):
            return "Message cannot be decrypted yet. Wait for the next block to be mined."
        
        # Get the next block and extract the private key from its hash
        next_block = self.chain[block_index + 1]
        next_block_header_string = json.dumps(next_block.header.to_dict(), sort_keys=True)
        # Calculate the private key from the next block's header
        private_key = int(hashlib.sha256(hashlib.sha256(next_block_header_string.encode()).digest()).hexdigest(), 16) % P
        
        # Decrypt the message using the private key
        try:
            decrypted_message = Transaction.decrypt_message(transaction.encrypted_message, private_key)
            return decrypted_message
        except Exception as e:
            return f"Decryption error: {str(e)}"
    
    def get_block_by_index(self, index: int) -> Optional[Dict]:
        """
        Get a block by its index in the blockchain.
        
        Args:
            index: Index of the block to retrieve
            
        Returns:
            Dictionary representation of the block if found, None otherwise
        """
        if 0 <= index < len(self.chain):
            return self.chain[index].to_dict()
        return None
    
    def get_chain(self) -> List[Dict]:
        """
        Get the full blockchain as a list of dictionaries.
        
        Returns:
            List of dictionaries, each representing a block
        """
        return [block.to_dict() for block in self.chain]

# Example usage demonstrating the time-release blockchain functionality
def main():
    """
    Main function to demonstrate the usage of the TimeReleaseBlockchain.
    
    This example:
    1. Creates a blockchain
    2. Adds time-locked messages
    3. Mines blocks
    4. Attempts to decrypt messages at different points
    """
    # Create a blockchain
    blockchain = TimeReleaseBlockchain()
    
    # Add some transactions
    # Message will be decryptable after 1 block
    tx1_id = blockchain.add_transaction("Alice", "Bob", "Hello Bob, this is a time-locked message!", 1)
    
    # Message will be decryptable after 2 blocks
    tx2_id = blockchain.add_transaction("Charlie", "Dave", "Secret message that should be revealed after 2 blocks", 2)
    
    # Mine a block (Block #1)
    print("\nMining block 1...")
    block1 = blockchain.mine_block()
    
    # Try to decrypt the messages (should fail for tx1_id since we need one more block)
    print("\nTrying to decrypt first message:")
    decrypted_message = blockchain.decrypt_message(1, tx1_id)
    print(f"Decrypted message: {decrypted_message}")
    
    # Should fail for tx2_id since we need two more blocks
    print("\nTrying to decrypt second message:")
    decrypted_message = blockchain.decrypt_message(1, tx2_id)
    print(f"Decrypted message: {decrypted_message}")
    
    # Mine another block (Block #2)
    print("\nMining block 2...")
    block2 = blockchain.mine_block()
    
    # Try to decrypt the messages again
    # Should succeed for tx1_id now
    print("\nTrying to decrypt first message:")
    decrypted_message = blockchain.decrypt_message(1, tx1_id)
    print(f"Decrypted message: {decrypted_message}")
    
    # Should still fail for tx2_id
    print("\nTrying to decrypt second message:")
    decrypted_message = blockchain.decrypt_message(1, tx2_id)
    print(f"Decrypted message: {decrypted_message}")
    
    # Add another transaction
    tx3_id = blockchain.add_transaction("Eve", "Adam", "Another time-locked message!", 1)
    
    # Mine another block (Block #3)
    print("\nMining block 3...")
    block3 = blockchain.mine_block()
    
    # Print the blockchain summary
    print("\nBlockchain:")
    for i, block in enumerate(blockchain.chain):
        print(f"Block {i}: {block.hash}")
        print(f"  Public Key: {block.header.public_key}")
        print(f"  Transactions: {len(block.transactions)}")
        # for j, transaction in enumerate(block.transactions):
        #     print(f"    Transaction_id {j}: {transaction.transaction_id}")
        #     print(f"    Transaction_sender_recipient {j}: {transaction.sender} -> {transaction.recipient}")
        #     print(f"    Transaction_encrypted_message {j}: {transaction.encrypted_message}")

if __name__ == "__main__":
    main()