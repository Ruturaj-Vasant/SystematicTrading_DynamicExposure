import yfinance as yf

def download_and_print_stock_data(stock_symbol):
    try:
        # Download stock data for the given symbol (hourly data for 2 years)
        stock_data = yf.download(stock_symbol, period="10y", interval="1d", auto_adjust=True)
        
        # Check if the data is empty
        if stock_data.empty:
            print(f"No data found for {stock_symbol}. Please check the symbol or date range.")
        else:
            # Print the first few rows of the downloaded data
            print("Downloaded Stock Data:")
            print(stock_data.head())
            
            # Calculate VWAP
            stock_data_with_vwap = vwap_execution(stock_data)
            print("\nStock Data with VWAP Execution:")
            print(stock_data_with_vwap[['Order']].head())  # Only showing 'Order' column for VWAP
    
    except Exception as e:
        print(f"An error occurred: {e}")

def vwap_execution(data, shares_to_execute=100):
    # Calculate the VWAP (Volume-Weighted Average Price)
    total_volume = data['Volume'].sum()
    
    # VWAP execution: Calculate the order (shares) based on the volume
    data['Order'] = (data['Volume'] / total_volume) * shares_to_execute
    return data

if __name__ == "__main__":
    # Example: Download and print AAPL data and calculate VWAP
    download_and_print_stock_data('AAPL')