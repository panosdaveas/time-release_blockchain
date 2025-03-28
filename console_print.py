from rich import print as rprint
from rich.pretty import pprint
from rich.pretty import Pretty
from rich.panel import Panel
from rich.console import Console, RenderableType
from rich.table import Column, Table
from rich.tree import Tree
from rich.layout import Layout
from rich import print_json
from rich.text import Text
from rich.containers import Lines, Renderables
import logging
from rich.logging import RichHandler
from rich.live import Live
from typing import List

# from rich import print_json

Max_String = 21
expand_all_flag = True
console = Console()
layout = Layout()
text = Text("foo")
renderables = Renderables([text])
log_messages = []

def header_grid(header) -> Table:
    grid = Table.grid(expand=True)
    grid.add_column()
    grid.add_column(justify="right", max_width=24, style="bold")
    grid.add_row("Index", f"{header.index}")
    grid.add_row("Previous Hash", f"{hex(int(header.prev_hash, 16))}")
    grid.add_row("Timestamp", f"{header.timestamp}")
    grid.add_row("Merkle Root", f"{hex(int(header.merkle_root, 16))}")
    grid.add_row("Nonce", f"{header.nonce}")
    grid.add_row("Public Key", f"({hex(header.public_key.h)}, {hex(header.public_key.g)}, {hex(header.public_key.p)})")
    grid.add_row("Public Key Length", f"{header.public_key_length} bits")
    return grid

def body_grid(transactions) -> Table:
    grid = Table.grid(expand=True)
    grid.add_column()
    grid.add_column(justify="right", max_width=16, style="bold", no_wrap=True)
    grid.add_row("Transactions count", f"{len(transactions)}")
    grid.add_row(f"Transactions" if len(transactions) > 0 else "[bold][red]No Transactions", "")
    for tx in transactions:
        grid.add_row("", "")
        grid.add_row("TxID", f"{hex(int(tx.transaction_id, 16))}")
        grid.add_row("Sender", f"{tx.sender}")
        grid.add_row("Receiver", f"{tx.recipient}")
        grid.add_row("Timestamp", f"{tx.timestamp}")
        grid.add_row("tx.encrypted_message", f"{tx.encrypted_message}")
    return grid

def block_grid(block) -> Table:
    panel_header = Panel(header_grid(block.header), title="[bold]Header", title_align="left")
    panel_body = Panel(body_grid(block.transactions), title="[bold]Body", title_align="left")
    grid = Table.grid(expand=True)
    grid.add_column()
    grid.add_row(panel_header)
    grid.add_row(panel_body)
    return grid

def blockchain_grid(blockchain) -> Table:
    grid = Table.grid(expand=True)
    grid.add_column()
    for block in blockchain:
        if block.header.index == 0:
            grid.add_row(Panel.fit(block_grid(block),title=f"BLOCK [bold red]{block.header.index}", subtitle="Genesis Block"))
        else:
            grid.add_row(Panel.fit(block_grid(block),title=f"BLOCK [bold red]{block.header.index}", subtitle=""))
    return grid

def current_block(blockchain) -> Panel:
    block = blockchain[-1]
    if block.header.index == 0:
        return Panel.fit(block_grid(block),title=f"BLOCK [bold red]{block.header.index}", subtitle="Genesis Block")
    else:
        return Panel.fit(block_grid(block),title=f"BLOCK [bold red]{block.header.index}", subtitle="Latest Block")

def tx_data(tx, index) -> dict:
    tx_data = {
        index: [
            f"ID: {tx.transaction_id[:8]}...{tx.transaction_id[-8:]}",
            {
                "Sender": f"{tx.sender}",
                "Receiver": f"{tx.recipient}",
                "Timestamp": f"{tx.timestamp}",
                "tx.encrypted_message": f"{tx.encrypted_message[:8]}...{tx.encrypted_message[-8:]}"
            }
        ]
    }
    return tx_data

def transactions_data(transactions) -> dict:
    data = dict()
    for i, tx in enumerate(transactions):
        data.update(tx_data(tx, i))
    return data

def tx_panel(blockchain) -> Panel:
    return Panel(Pretty(transactions_data(blockchain[-1].transactions), indent_guides=True), title="Transactions", expand=True)

def print_blockchain(blockchain):
    rprint(blockchain_grid(blockchain))

def comparison(renderable1: RenderableType, renderable2: RenderableType) -> Table:
        table = Table(show_header=False, pad_edge=False, box=None, expand=True)
        table.add_column("1", ratio=1)
        table.add_column("2", ratio=2)
        table.add_row(renderable1, renderable2)
        return table

def update_logs(str) -> Text:
    text = Text("Logging:")
    # text.stylize("bold magenta", 0, 6)
    text.append(str, style="bold red")
    return text

def upper_layout_content():
    return Panel(update_logs("logging"), expand=True)

def table_content() -> Table:
    table = Table.grid(padding=1, pad_edge=True)
    table.add_column("Feature", no_wrap=True, justify="center", style="bold red")
    table.add_column("Demonstration")
    table.add_row(
        "Syntax\nhighlighting\n&\npretty\nprinting",
        comparison(
        ),
    )
    return table

def layout_content() -> Layout:
    layout.split_column(
    Layout("", name="upper"),
    Layout("", name="lower")
    )
    layout["lower"].split_row(
    Layout("", name="left"),
    Layout("", name="right", ratio=2),
    )
    return layout

def update_logs(new_message: str = None) -> Text:
    """
    Update and retrieve log messages.
    
    Args:
        new_message: Optional new message to add to logs
    
    Returns:
        Rich Text object with formatted log messages
    """
    global log_messages
    
    # Add new message if provided
    if new_message:
        log_messages.append(new_message)
    
    # Keep only the last 10 log messages
    if len(log_messages) > 10:
        log_messages = log_messages[-10:]
    
    # Create a Text object with log messages
    log_text = Text()
    for msg in log_messages:
        log_text.append(msg + "\n", style="dim")
    
    return log_text

def log_message(message: str):
    """
    Log a message to be displayed in the upper layout.
    
    Args:
        message: Message to log
    """
    update_logs(message)

# print the blockchain from trb.py file in a pretty format
def print_layout(blockchain): 
    # rprint(blockchain_grid(blockchain))
    def update_display():
        upper_layout = Panel(update_logs(),title="Time-Release Blockchain", subtitle="beta version", expand=True)
        layout = layout_content()
        layout["upper"].update(upper_layout)
        layout["lower"]["left"].update((current_block(blockchain))),
        layout["lower"]["right"].update((tx_panel(blockchain))),
        return layout
        # rprint(layout)
    # rprint(comparison(current_block(blockchain), tx))

   # Use Live to update the display dynamically
    live = Live(update_display(), refresh_per_second=4, vertical_overflow="crop")
    live.start()

    def update_callback(updated_blockchain: List):
        """
        Update the live display with the new blockchain state
        
        Args:
            updated_blockchain: The current state of the blockchain
        """
        global blockchain  # Use global to modify the reference
        blockchain = updated_blockchain
        live.update(update_display())

    return update_callback

def main():
    pass

if __name__ == "__main__":
    main()