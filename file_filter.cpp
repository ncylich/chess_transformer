#include <iostream>
#include <fstream>
#include <string>

using namespace std;

const string INFILE = "lichess_db_standard_rated_2023-11.pgn";
const string OUTFILE = "filtered_games.pgn";

int get_num(string &s) {
    int num = 0;
    for (char ch : s) {
        if (ch >= '0' && ch <= '9') {
            num *= 10;
            num += ch - '0';
        }
    }
    return num;
}

int main(int argc, char* argv[]) {
    const string pwd = "E:\\Python\\chess_transformer\\";
    string totInFile, totOutFile;
    int minelo = 2000;
    int maxelo = 99999;
    if (argc == 1) {
        totInFile = pwd + INFILE;
        totOutFile = pwd + OUTFILE;
    } else if (argc == 4) {
        minelo = stoi(argv[1]);
        totInFile = pwd + argv[2];
        totOutFile = pwd + argv[3];
    } else if (argc == 5) {
        minelo = stoi(argv[1]);
        maxelo = stoi(argv[2]);
        totInFile = pwd + argv[3];
        totOutFile = pwd + argv[4];
    }else {
        cerr << "Usage: ./file_filter [input_file] [output_file]" << endl;
        return 1;
    }

    ifstream inFile(totInFile);
    ofstream outFile(totOutFile);

    if (!inFile.is_open() || !outFile.is_open()) {
        cerr << "Error opening files" << endl;
        return 1;
    }

    string curr_game, curr_line, event;
    int elo = 0;
    int count = 0;

    while (getline(inFile, curr_line)) {
        curr_game += curr_line + "\n";

        if (curr_line.substr(0, 2) == "1.") {
            if (event.find("Blitz") == string::npos && event.find("Bullet") == string::npos && elo >= minelo && elo < maxelo) {
                outFile << curr_game << endl;
                count++;
                if (count % 1000 == 0) {
                    cout << count << endl;
                }
            }
            // Reset for next game
            curr_game = "";
            event = "";
            elo = maxelo;
        } else {
            if (curr_line.find("Event") != string::npos) {
                event = curr_line;
            }
            // handles both white and black elos at the same time
            else if (curr_line.find("Elo") != string::npos) {
                // if elo are not smaller than maxelo, then elo == maxelo and game filters out
                elo = min(get_num(curr_line), elo);
            }
        }
    }

    inFile.close();
    outFile.close();
    cout << "Total games processed: " << count << endl;
    cout << "Output file: " << totOutFile << endl;
    return 0;
}
