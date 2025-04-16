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
from rich import box
import time
from rich.progress import track
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)


# from rich import print_json

Max_String = 21
expand_all_flag = True
console = Console()
layout = Layout()
text = Text("foo")
renderables = Renderables([text])
log_messages = []
transactions = []


def block_header_grid(header) -> Table:
    grid = Table.grid(expand=True)
    grid.add_column()
    grid.add_column(justify="right", style="bold")
    grid.add_row("Index", f"{header.index}")
    grid.add_row("Previous Hash", f"{hex(int(header.prev_hash, 16))}")
    grid.add_row("Timestamp", f"{header.timestamp}")
    grid.add_row("Merkle Root", f"{hex(int(header.merkle_root, 16))}")
    grid.add_row("Nonce", f"{header.nonce}")
    grid.add_row(
        "Public Key", f"({hex(header.public_key.h)}, {hex(header.public_key.g)}, {hex(header.public_key.p)})")
    grid.add_row("Public Key Length", f"{header.public_key_length} bits")
    return grid


def block_body_grid(transactions) -> Table:
    grid = Table.grid(expand=True)
    grid.add_column()
    grid.add_column(justify="right", style="bold", no_wrap=True)
    grid.add_row("Transactions count", f"{len(transactions)}")
    return grid


def block_grid(block) -> Table:
    panel_header = Panel(block_header_grid(block.header),
                         title="[bold]Header", title_align="left")
    panel_body = Panel(block_body_grid(block.transactions),
                       title="[bold]Body", title_align="left")
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
            grid.add_row(Panel.fit(block_grid(
                block), title=f"BLOCK [bold red]{block.header.index}", subtitle="Genesis Block"))
        else:
            grid.add_row(Panel.fit(block_grid(
                block), title=f"BLOCK [bold red]{block.header.index}", subtitle=""))
    return grid


def current_block(blockchain) -> Panel:
    block = blockchain[-1]
    if block.header.index == 0:
        return Panel(block_grid(block), title=f"BLOCK [bold red]{block.header.index}", subtitle="Genesis Block")
    else:
        return Panel(block_grid(block), title=f"BLOCK [bold red]{block.header.index}", subtitle="")


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
    return Panel(Pretty(transactions_data(blockchain[-1].transactions), indent_guides=True), title="Transactions", expand=True, border_style="none")


def print_blockchain(blockchain):
    rprint(blockchain_grid(blockchain))


def comparison3(renderable1: RenderableType, renderable2: RenderableType, rendable3: RenderableType) -> Table:
    table = Table(show_header=False, pad_edge=False, box=None, expand=True)
    table.add_column("1", style="bold red")
    table.add_column("2", ratio=1)
    table.add_column("3", ratio=1)
    table.add_row(renderable1, renderable2, rendable3)
    return table


def comparison2(renderable1: RenderableType, renderable2: RenderableType) -> Table:
    table = Table(show_header=False, pad_edge=False, box=None, expand=True)
    table.add_column("1", ratio=1, style="bold red")
    table.add_column("2", ratio=2)
    table.add_row(renderable1, renderable2)
    return table


table = Table(
    show_edge=False,
    show_header=True,
    expand=True,
    row_styles=["none", "dim"],
    box=box.SIMPLE,
)
table.add_column("TxID", justify="left", style="bold red", no_wrap=True)
table.add_column("From", justify="left", style="bold red", no_wrap=True)
table.add_column("", justify="left", style="bold red", no_wrap=True)
table.add_column("To", justify="left", style="bold red", no_wrap=True)
table.add_column("Payload", justify="left", style="bold red", no_wrap=True)

def transactionsTable(transaction: dict = None, decrypted_message: str = None) -> Table:
    global transactions


    if transaction:
        transaction.decrypted_message = decrypted_message
        transactions.append(transaction)
    
    if len(transactions) > 10:
        transactions = transactions[-10:]

    for tx in reversed(transactions):
        table.add_row(
            f"{hex(int(tx.transaction_id, 16))}"[:8],
            f"{tx.sender}",
            f"->",
            f"{tx.recipient}",
            f"{tx.decrypted_message}"
            # f"{tx.encrypted_message[:8]}...{tx.encrypted_message[-8:]}"
        )
    return table

def log_transaction(tx, decrypted_message):
    transactionsTable(tx, decrypted_message)

def keyPairs(blockchain) -> Table:
    table = Table(
        # title="Key Pairs",
        show_edge=False,
        show_header=True,
        expand=False,
        # row_styles=["none", "dim"],
        box=box.SIMPLE,
    )
    table.add_column("Public", justify="left", style="bold red", no_wrap=True)
    table.add_column("Private", justify="left", style="bold red", no_wrap=True)
    if len(blockchain) > 1:
        for i in range(len(blockchain) - 1):
            style = None if i == 0 else "dim"
            table.add_row(
                f"{hex(blockchain[i].header.public_key.h)}",
                f"{hex(blockchain[i+1].header.nonce)}",
                style=style
            )
        # table.add_row(
        #     f"{blockchain[-2].header.public_key.h}",
        #     f"{blockchain[-1].header.nonce}"
        # )
    return table


def upper_layout_content():
    return Panel(update_logs("logging"), expand=True)


def table_content() -> Table:
    table = Table.grid(padding=1, pad_edge=True)
    table.add_column("Feature", no_wrap=True,
                     justify="center", style="bold red")
    table.add_column("Demonstration")
    table.add_row(
        "Syntax\nhighlighting\n&\npretty\nprinting",
    )
    return table


def layout_content() -> Layout:
    layout.split_column(
        Layout("", name="upper"),
        Layout("", name="lower")
    )
    layout["lower"].split_row(
        Layout("", name="current_block"),
        Layout("", name="current_transactions", ratio=2),
    )
    layout["upper"].split_row(
        Layout("", name="progress", ratio=1),
        Layout("", name="logs", ratio=1),
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
    if len(log_messages) > 16:
        log_messages = log_messages[-16:]
    
    # Create a Text object with log messages
    log_text = Text()
    for i, msg in enumerate(log_messages):
        if i == len(log_messages) - 1:  # Last message
            log_text.append(msg + "\n", style="bold red")
        else:
            log_text.append(msg + "\n", style="dim")
    return log_text


def log_message(message: str):
    """
    Log a message to be displayed in the upper layout.

    Args:
        message: Message to log
    """
    update_logs(message)


def create_mining_progress(description: str = "Mining block...", total: float = 1.0) -> Progress:
    """
    Create a mining progress bar for the upper layout.

    Args:
        description: Description of the mining task
        total: Total work units for the progress bar

    Returns:
        Rich Progress object
    """
    return Progress(
        SpinnerColumn(),
        # BarColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True  # Removes the progress bar after completion
    )


def update_mining_progress(progress: Progress, task_id: int, advance: float = 0.5) -> None:
    """
    Update the mining progress bar.

    Args:
        progress: Rich Progress object
        task_id: ID of the task to update
        advance: Amount to advance the progress bar
    """
    progress.update(task_id, advance=advance)


def mining_layout(blockchain=None, mining_progress: Progress = None) -> Table:
    """
    Create a layout with mining progress and logs.

    Args:
        blockchain: Optional blockchain for current block display
        mining_progress: Optional Progress object for mining

    Returns:
        Rich Layout object
    """
    grid = Table.grid(expand=True, padding=1, pad_edge=True)
    grid.add_column(justify="left")

    # Update progress layout if a progress bar is provided
    grid.add_row(
        Panel(mining_progress if mining_progress else "",
              title="Mining Progress", border_style="green", expand=True)
    )

    # Update logs layout with update_logs()
    grid.add_row(Panel(update_logs(), height=12, title="Logs",
                 border_style="blue", expand=True))

    # Update current block if blockchain is provided
    if blockchain and blockchain:
        grid.add_row(
            comparison3(
                "Latest\nmined block",
                current_block(blockchain),
                keyPairs(blockchain),
                # Pretty(transactions_data(blockchain[-1].transactions), indent_guides=True)
            )
        )
        # grid.add_row(transactionsTable(blockchain[-1].transactions))

    return grid

# print the blockchain from trb.py file in a pretty format


def print_layout(blockchain):
    console.clear()
    # rprint(blockchain_grid(blockchain))

    def update_display(mining_progress: Progress = None):
        return mining_layout(blockchain, mining_progress)

    # Use Live to update the display dynamically
    live = Live(update_display(), refresh_per_second=4,
                vertical_overflow="crop")
    live.start()

    def update_callback(updated_blockchain: List, mining_progress: Progress = None):
        """
        Update the live display with the new blockchain state

        Args:
            updated_blockchain: The current state of the blockchain
            mining_progress: Optional Progress object for mining
        """
        global blockchain  # Use global to modify the reference
        blockchain = updated_blockchain
        live.update(update_logs())
        live.update(update_display(mining_progress))
        # wait until the live display is updated
        wait = 1
        time.sleep(wait)

    return update_callback


def make_layout(blockchain=None, mining_progress: Progress = None) -> Layout:
    """Define the layout."""
    layout = Layout(name="root")

    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=10),
    )
    layout["main"].split_row(
        Layout(name="side"),
        Layout(name="body", ratio=2, minimum_size=30),
    )
    layout["side"].split(Layout(name="box1", ratio=2), Layout(name="box2"))

    layout["header"].update(Panel(
        mining_progress, title="Time Release Blockchain", border_style="green", expand=True))
    layout["body"].update(
        Panel(update_logs(), title="Logs", border_style="blue", expand=True))
    layout["box2"].update(Panel(comparison2("Generated\nkey pairs", keyPairs(
        blockchain)), border_style="none", expand=True))
    layout["box1"].update(current_block(blockchain))
    # layout["box1"].update(Panel(current_block(blockchain), title="Latest Block", border_style="white", expand=True))
    layout["footer"].update(Panel(transactionsTable(), title="Transactions", border_style="red", expand=True))

    return layout


def main():
    # console.clear()
    # layout = layout_content()
    layout = make_layout()
    layout["header"].update(
        Panel("", title="Time Release Blockchain", border_style="green", expand=True))
    layout["body"].update(
        Panel("", title="Logs", border_style="blue", expand=True))
    layout["box2"].update(
        Panel("", title="Key Pairs", border_style="green", expand=True))
    layout["box1"].update(Panel("", title="Latest Block",
                          border_style="white", expand=True))
    layout["footer"].update(
        Panel("", title="Transactions", border_style="red", expand=True))
    rprint(layout)

    # pass


def display(blockchain):

    def update_display(mining_progress: Progress = None):
        return make_layout(blockchain, mining_progress)

    # Use Live to update the display dynamically
    live = Live(update_display(), refresh_per_second=4,
                vertical_overflow="crop")
    live.start()

    def update_callback(updated_blockchain: List, mining_progress: Progress = None):
        """
        Update the live display with the new blockchain state

        Args:
            updated_blockchain: The current state of the blockchain
            mining_progress: Optional Progress object for mining
        """
        global blockchain  # Use global to modify the reference
        blockchain = updated_blockchain
        # live.update(transactionsTable())
        live.update(update_display(mining_progress))
        # wait until the live display is updated
        # wait = 1
        # time.sleep(wait)

    return update_callback


if __name__ == "__main__":
    main()
