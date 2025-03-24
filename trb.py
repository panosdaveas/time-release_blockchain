"""
Time-Release Blockchain Implementation with ElGamal Encryption

This code provides a time-release blockchain implementation using ElGamal encryption.
"""
import hashlib
import time
import json
import random
import sys
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# Import ElGamal encryption module
import elgamal

class Transaction:
    def __init__(self, sender: str, recipient: str, message: str, public_key: elgamal.PublicKey):
        """
        Create a transaction with an encrypted message using ElGamal encryption.
        
        Args:
            sender: Sender's identifier
            recipient: Recipient's identifier
            message: Message to encrypt
            public_key: Public key to encrypt the message with
        """
        self.sender = sender
        self.recipient = recipient
        self.timestamp = time.time()  # Record current time as transaction timestamp
        
        # Encrypt the message with the provided public key
        self.encrypted_message = elgamal.encrypt(public_key, message)
        
        # Generate a unique transaction ID based on transaction details
        self.transaction_id = self.calculate_hash()
    
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
    
    @staticmethod
    def decrypt_message(encrypted_message: str, private_key: elgamal.PrivateKey) -> str:
        """
        Decrypt a message using the provided private key.
        
        Args:
            encrypted_message: Encrypted message string
            private_key: Private key for decryption
            
        Returns:
            Decrypted message as a string
        """
        return elgamal.decrypt(private_key, encrypted_message)
    
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
    public_key: elgamal.PublicKey = None  # Public key for this block
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
            "public_key_h": self.public_key.h,
            "public_key_g": self.public_key.g,
            "public_key_p": self.public_key.p,
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
    def __init__(self, seed: int = None, num_bits: int = 20):
        """
        Initialize the blockchain with a genesis block.
        
        Args:
            seed: Random seed for key generation
            num_bits: Number of bits for prime generation
        """
        self.chain = []  # List to store all blocks
        self.pending_transactions = []  # Transactions waiting to be included in a block
        
        # Seed for reproducibility
        self.seed = seed or random.randint(1, sys.maxsize)
        self.num_bits = num_bits
        
        # Create the genesis block
        self.create_genesis_block()
    
    def create_genesis_block(self):
        """
        Create the genesis block with an initial public key.
        """
        # Generate initial public key
        genesis_public_key = elgamal.generate_keys(seed=self.seed, iNumBits=self.num_bits)[0]
        
        # Create genesis block with index 0 and empty previous hash
        genesis_block = Block(0, "0" * 64, [])
        genesis_block.header.public_key = genesis_public_key
        genesis_block.hash = genesis_block.calculate_hash()
        
        # Add to chain
        self.chain.append(genesis_block)
        print(f"Genesis block created with public key h: {hex(genesis_public_key.h)}")
    
    def generate_next_public_key(self, previous_public_key: tuple[int, int, int]) -> elgamal.PublicKey:
        """
        Generate next public key with a new seed.
        
        Returns:
            New ElGamal public key
        """
        # Use the current time as a seed to generate a new key
        # This ensures different keys for each block while maintaining determinism
        
        # new_seed = int(time.time() * 1000000) + random.randint(1, 1000000)
        new_seed = int(previous_public_key.p + previous_public_key.g + previous_public_key.h)
        return elgamal.generate_keys(seed=new_seed, iNumBits=self.num_bits)[0]
    
    def add_transaction(self, sender: str, recipient: str, message: str, blocks_ahead: int = 1):
        """
        Add a new transaction to the pending transactions list.
        
        Args:
            sender: Sender's identifier
            recipient: Recipient's identifier
            message: Message to encrypt
            blocks_ahead: Number of blocks to wait before the message can be decrypted
            
        Returns:
            Transaction ID of the created transaction
        """
        # For time-locked messages, calculate future public keys
        future_public_keys = self.calculate_future_public_keys(blocks_ahead)
        
        if not future_public_keys:
            raise ValueError("Failed to calculate future public keys")
        
        # Encrypt the message with the last future public key
        final_tx = Transaction(sender, recipient, message, future_public_keys[-1])
        self.pending_transactions.append(final_tx)
        
        print(f"Transaction added: Message will be decryptable after {blocks_ahead} blocks")
        return final_tx.transaction_id
    
    def calculate_future_public_keys(self, blocks_ahead: int) -> List[elgamal.PublicKey]:
        """
        Calculate public keys for future blocks.
        
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
        # Generate future public keys
        for _ in range(blocks_ahead):
            # Generate a new public key
            future_public_key = self.generate_next_public_key(current_public_key)
            print(f"Future public key h: {hex(future_public_key.h)}")
            future_public_keys.append(future_public_key)
            current_public_key = future_public_key
        
        return future_public_keys
    
    def mine_block(self) -> Block:
        """
        Mine a new block with the pending transactions.
        
        The mining process involves finding a nonce that satisfies ElGamal key verification.
        
        Returns:
            The newly mined block
        """
        if not self.chain:
            raise ValueError("Cannot mine block: blockchain is empty")
        
        # Get the last block and its public key
        last_block = self.chain[-1]
        
        # Create a new block with current pending transactions
        new_block = Block(
            index=len(self.chain),
            prev_hash=last_block.hash,
            transactions=self.pending_transactions.copy()
        )
        
        # Generate public key for this block
        new_public_key = self.generate_next_public_key(last_block.header.public_key)
        new_block.header.public_key = new_public_key
        
        # Mining process: find a nonce that produces a hash satisfying the ElGamal key verification
        print("Mining block...")
        start_time = time.time()
        
        while True:
            # Try a new nonce
            new_block.header.nonce += 1
            
            # Calculate header string
            header_string = json.dumps(new_block.header.to_dict(), sort_keys=True)
            
            # Calculate double SHA-256 hash
            hash_bytes = hashlib.sha256(hashlib.sha256(header_string.encode()).digest()).digest()
            hash_int = int.from_bytes(hash_bytes, byteorder='big')
            
            # Verify the key esing ElGamal verification
            if new_block.header.public_key.h == elgamal.modexp(new_block.header.public_key.g, hash_int, new_block.header.public_key.p):
                private_key = elgamal.PrivateKey(
                    p=new_block.header.public_key.p, 
                    g=new_block.header.public_key.g, 
                    x=hash_int, 
                    iNumBits=new_block.header.public_key_length
                )
                block_hash = hash_bytes.hex()
                break
        
        mining_time = time.time() - start_time
        print(f"Block mined in {mining_time:.2f} seconds with nonce: {new_block.header.nonce}")
        print(f"Private key for this block derived from block hash: {private_key.p, private_key.g, private_key.x}")
        
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
        if block_index + 1 >= len(self.chain):
            return "Message cannot be decrypted yet. Wait for the next block to be mined."
        
        # extract the private key from the last mined block
        current_block = self.chain[block_index]
        current_block_header_string = json.dumps(current_block.header.to_dict(), sort_keys=True)
        current_block_hash_bytes = hashlib.sha256(hashlib.sha256(current_block_header_string.encode()).digest()).digest()
        current_block_hash_int = int.from_bytes(current_block_hash_bytes, byteorder='big')
        
        # Create private key from the current block's hash
        private_key = elgamal.PrivateKey(
            p=current_block.header.public_key.p, 
            g=current_block.header.public_key.g, 
            x=current_block_hash_int, 
            iNumBits=current_block.header.public_key_length
        )
        
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
    blockchain = TimeReleaseBlockchain(seed=833050814021254693158343911234888353695402778102174580258852673738983005, num_bits=20)
    
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
    tx3_id = blockchain.add_transaction("Eve", "Adam", "Another time locked message", 1)
    
    # Mine another block (Block #3)
    print("\nMining block 3...")
    block3 = blockchain.mine_block()

     # Should succeed for tx2_id
    print("\nTrying to decrypt second message:")
    decrypted_message = blockchain.decrypt_message(1, tx2_id)
    print(f"Decrypted message: {decrypted_message}")
    
    # Print the blockchain summary
    print("\nBlockchain:")
    for i, block in enumerate(blockchain.chain):
        print(f"Block {i}: {block.hash}")
        print(f"  Public Key h: {hex(block.header.public_key.h)}")
        print(f"  Transactions: {len(block.transactions)}")
        # for tx in block.transactions:
        #     print(f"    {tx.sender} -> {tx.recipient}: {tx.encrypted_message}")

if __name__ == "__main__":
    main()