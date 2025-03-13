from dash import html

def create_header_cell(text, **kwargs):
    """Create a table header cell with proper formatting"""
    return html.Th(text, **kwargs)

def create_table_cell(content, **kwargs):
    """Create a table cell with proper formatting"""
    if isinstance(content, dict) and 'name' in content:
        # Handle dict format by extracting name
        return html.Td(content['name'], **kwargs)
    return html.Td(str(content), **kwargs)

def create_table_row(cells, is_header=False):
    """Create a table row with proper cell components"""
    cell_func = create_header_cell if is_header else create_table_cell
    return html.Tr([cell_func(cell) for cell in cells])

def create_table(headers, rows, id=None):
    """Create a complete table with proper structure"""
    return html.Table([
        html.Thead(create_table_row(headers, is_header=True)),
        html.Tbody([create_table_row(row) for row in rows])
    ], id=id)
