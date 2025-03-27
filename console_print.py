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

# from rich import print_json

Max_String = 21
expand_all_flag = True
console = Console()
layout = Layout()
text = Text("foo")
renderables = Renderables([text])

def header_grid(header):
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

def body_grid(transactions):
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

def block_grid(block):
    panel_header = Panel(header_grid(block.header), title="[bold]Header", title_align="left")
    panel_body = Panel(body_grid(block.transactions), title="[bold]Body", title_align="left")
    grid = Table.grid(expand=True)
    grid.add_column()
    grid.add_row(panel_header)
    grid.add_row(panel_body)
    return grid

def blockchain_grid(blockchain):
    grid = Table.grid(expand=True)
    grid.add_column()
    for block in blockchain:
        if block.header.index == 0:
            grid.add_row(Panel.fit(block_grid(block),title=f"BLOCK [bold red]{block.header.index}", subtitle="Genesis Block"))
        else:
            grid.add_row(Panel.fit(block_grid(block),title=f"BLOCK [bold red]{block.header.index}", subtitle=""))
    return grid

def current_block(blockchain):
    block = blockchain[-1]
    if block.header.index == 0:
        return Panel.fit(block_grid(block),title=f"BLOCK [bold red]{block.header.index}", subtitle="Genesis Block")
    else:
        return Panel.fit(block_grid(block),title=f"BLOCK [bold red]{block.header.index}", subtitle="Latest Block")

def tx_data(tx, index):
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

def transactions_data(transactions):
    data = dict()
    for i, tx in enumerate(transactions):
        data.update(tx_data(tx, i))
    return data

def print_blockchain(blockchain):
    rprint(blockchain_grid(blockchain))

def comparison(renderable1: RenderableType, renderable2: RenderableType) -> Table:
        table = Table(show_header=False, pad_edge=False, box=None, expand=True)
        table.add_column("1", ratio=1)
        table.add_column("2", ratio=2)
        table.add_row(renderable1, renderable2)
        return table

def update_logs(str):
    text = Text("Logging:")
    # text.stylize("bold magenta", 0, 6)
    text.append(str, style="bold red")
    return text

def upper_layout_content():
    return Panel(update_logs("logging"), expand=True)

def table_content():
    table = Table.grid(padding=1, pad_edge=True)
    table.add_column("Feature", no_wrap=True, justify="center", style="bold red")
    table.add_column("Demonstration")
    table.add_row(
        "Syntax\nhighlighting\n&\npretty\nprinting",
        comparison(
            Syntax(code, "python3", line_numbers=True, indent_guides=True),
            Pretty(transactions_data(blo), indent_guides=True),
        ),
    )
    return table

# print the blockchain from trb.py file in a pretty format
def print_layout(blockchain): 
    rprint(blockchain_grid(blockchain))
    upper_layout = Panel(update_logs(str="Logging_data"),title="Time-Release Blockchain", subtitle="beta version", expand=True)
    tx = Panel(Pretty(transactions_data(blockchain[1].transactions), indent_guides=True), title="Transactions", expand=True)
    layout.split_column(
    Layout(upper_layout, name="upper"),
    Layout("", name="lower")
    )
    # layout["upper"].update(update_logs(""))
    layout["lower"].split_row(
    Layout(current_block(blockchain), name="left"),
    Layout(tx, name="right"),
    )
    layout["lower"]["right"].ratio = 2
    rprint(layout)
    # rprint(comparison(current_block(blockchain), tx))

def main():
    pass

if __name__ == "__main__":
    main()