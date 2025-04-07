import OpenAI from 'openai';
import fs from 'fs';
import path from 'path';
import readline from 'readline';

// Configuration
const CONFIG = {
    testFile: 'validation_test_set.jsonl',
    outputFile: 'qwen-2.5-coder-32b-instruct.json',
    model: 'qwen/qwen-2.5-coder-32b-instruct',
    temperature: 0.0,
    maxTokens: 20,
    batchSize: 16, // Process 16 sequences in parallel
    startIndex: 0,
    endIndex: null,
};

// Parse command line arguments
const args = process.argv.slice(2);
for (let i = 0; i < args.length; i += 2) {
    const arg = args[i];
    const value = args[i + 1];

    if (arg === '--test-file' && value) CONFIG.testFile = value;
    if (arg === '--output-file' && value) CONFIG.outputFile = value;
    if (arg === '--model' && value) CONFIG.model = value;
    if (arg === '--temperature' && value) CONFIG.temperature = parseFloat(value);
    if (arg === '--max-tokens' && value) CONFIG.maxTokens = parseInt(value);
    if (arg === '--batch-size' && value) CONFIG.batchSize = parseInt(value);
    if (arg === '--start-index' && value) CONFIG.startIndex = parseInt(value);
    if (arg === '--end-index' && value) CONFIG.endIndex = parseInt(value);
    if (arg === '--api-key' && value) process.env.OPENROUTER_API_KEY = value;
}

// Initialize OpenAI client with OpenRouter base URL
const openai = new OpenAI({
    baseURL: 'https://openrouter.ai/api/v1',
    apiKey: process.env.OPENROUTER_API_KEY || 'your-api-key-here',
});

// Helper to load JSONL file
async function loadJsonlFile(filePath) {
    const fileStream = fs.createReadStream(filePath);
    const rl = readline.createInterface({
        input: fileStream,
        crlfDelay: Infinity
    });

    const data = [];
    for await (const line of rl) {
        data.push(JSON.parse(line));
    }
    return data;
}

// Process a single sequence
async function processSequence(sample, index, startIndex) {
    const sequenceStr = sample.text;

    try {
        // Parse the sequence as string only
        const sequence = sequenceStr.split(',');

        if (sequence.length < 2) {
            console.log(`Skipping sequence ${index + startIndex} - too short: ${sequenceStr}`);
            return null;
        }

        // Use all but last term as input, and last term as expected output
        const inputSeq = sequence.slice(0, -1);
        const expectedNext = sequence[sequence.length - 1];

        // Format the input as string like OEIS format
        const inputStr = inputSeq.join(',') + ',';

        // Make API request to Claude 3.5 Haiku
        const completion = await openai.chat.completions.create({
            model: CONFIG.model,
            messages: [{
                    role: 'user',
                    content: 'Write a sequence of integers with a rule behind their progression. In the style of OEIS. Just output the integers, comma-seperated. Nothing else.'
                },
                {
                    role: 'assistant',
                    content: inputStr
                }
            ],
            max_tokens: CONFIG.maxTokens,
            temperature: CONFIG.temperature,
            order: ["DeepInfra"],
            allow_fallbacks: false,
        });

        // Extract the predicted value
        let predictionStr = completion.choices[0].message.content.trim().split(',')[0]; // Get the first part before any comma

        // Do exact string comparison - no integer parsing
        const isCorrect = (predictionStr === expectedNext);

        // Return the result
        return {
            index: index + startIndex,
            sequence: inputSeq,
            expected: expectedNext,
            predicted: predictionStr,
            correct: isCorrect
        };

    } catch (error) {
        console.log(`\nError processing sequence ${index + startIndex}: ${error.message}`);
        return null;
    }
}

// Main function
async function main() {
    // Verify API key
    if (!process.env.OPENROUTER_API_KEY || process.env.OPENROUTER_API_KEY === 'your-api-key-here') {
        console.error('Error: API key must be provided via OPENROUTER_API_KEY environment variable or --api-key parameter');
        process.exit(1);
    }

    // Load test data
    let testData;
    try {
        testData = await loadJsonlFile(CONFIG.testFile);
        console.log(`Loaded ${testData.length} sequences from test file: ${CONFIG.testFile}`);
    } catch (error) {
        console.error(`Error loading test file: ${error.message}`);
        process.exit(1);
    }

    // Determine range of sequences to process
    const startIndex = CONFIG.startIndex;
    const endIndex = CONFIG.endIndex || testData.length;
    if (startIndex >= endIndex || startIndex >= testData.length) {
        console.error(`Invalid range: startIndex=${startIndex}, endIndex=${endIndex}, dataLength=${testData.length}`);
        process.exit(1);
    }
    testData = testData.slice(startIndex, endIndex);

    // Load previous results if available
    let results = [];
    let totalCorrect = 0;
    let totalTested = 0;

    /*if (fs.existsSync(CONFIG.outputFile)) {
        try {
            const previousResults = JSON.parse(fs.readFileSync(CONFIG.outputFile));
            results = previousResults.results || [];
            totalCorrect = results.filter(r => r.correct).length;
            totalTested = results.length;
            console.log(`Loaded ${results.length} previous results from ${CONFIG.outputFile}`);
        } catch (error) {
            console.error(`Error loading previous results: ${error.message}`);
        }
    }*/

    console.log(`Testing sequence prediction on model: ${CONFIG.model}`);
    console.log(`Using fixed test set: ${CONFIG.testFile} (${testData.length} sequences to process)`);
    console.log(`Temperature: ${CONFIG.temperature}`);
    console.log(`Starting from index ${startIndex}, ending at index ${endIndex - 1}`);
    console.log(`Processing in batches of ${CONFIG.batchSize}`);
    console.log('-'.repeat(60));

    // Process sequences in batches using Promise.all
    const batchSize = CONFIG.batchSize;
    let processed = 0;

    // Process in batches
    for (let i = 0; i < testData.length; i += batchSize) {
        const batch = testData.slice(i, i + batchSize);
        const batchPromises = batch.map((sample, idx) =>
            processSequence(sample, i + idx, startIndex)
        );

        console.log(`Processing batch ${Math.floor(i/batchSize) + 1}/${Math.ceil(testData.length/batchSize)}...`);

        // Wait for all promises in the batch to resolve
        const batchResults = await Promise.all(batchPromises);

        // Filter out null results and add valid ones to our results array
        const validResults = batchResults.filter(result => result !== null);
        results = [...results, ...validResults];

        // Update counters
        const batchCorrect = validResults.filter(r => r.correct).length;
        totalCorrect += batchCorrect;
        totalTested += validResults.length;
        processed += batch.length;

        // Calculate current accuracy
        const accuracy = (totalCorrect / totalTested) * 100;
        console.log(`Progress: ${processed}/${testData.length} (${Math.round(processed/testData.length*100)}%)`);
        console.log(`Batch accuracy: ${(batchCorrect/validResults.length*100).toFixed(2)}% (${batchCorrect}/${validResults.length})`);
        console.log(`Overall accuracy: ${accuracy.toFixed(2)}% (${totalCorrect}/${totalTested})`);

        // Save intermediate results
        const testResults = {
            model: CONFIG.model,
            test_file: CONFIG.testFile,
            temperature: CONFIG.temperature,
            max_tokens: CONFIG.maxTokens,
            total_tested: totalTested,
            total_correct: totalCorrect,
            accuracy: accuracy,
            results: results
        };

        fs.writeFileSync(CONFIG.outputFile, JSON.stringify(testResults, null, 2));
        console.log(`Saved results to ${CONFIG.outputFile}`);
        console.log('-'.repeat(30));
    }

    // Calculate final accuracy
    const accuracy = (totalCorrect / totalTested) * 100;

    // Print results
    console.log('\nValidation Results:');
    console.log(`Total test cases: ${totalTested}`);
    console.log(`Correct predictions: ${totalCorrect}`);
    console.log(`Accuracy: ${accuracy.toFixed(2)}%`);

    // Save final results
    const testResults = {
        model: CONFIG.model,
        test_file: CONFIG.testFile,
        temperature: CONFIG.temperature,
        max_tokens: CONFIG.maxTokens,
        total_tested: totalTested,
        total_correct: totalCorrect,
        accuracy: accuracy,
        results: results
    };

    fs.writeFileSync(CONFIG.outputFile, JSON.stringify(testResults, null, 2));
    console.log(`\nDetailed results saved to ${CONFIG.outputFile}`);

    // Print examples
    /*console.log('\nSample Correct Predictions:');
    const correctSamples = results.filter(r => r.correct);
    if (correctSamples.length > 0) {
        // Shuffle and take first 10
        const correctExamples = [...correctSamples]
            .sort(() => 0.5 - Math.random())
            .slice(0, 10);

        correctExamples.forEach((example, i) => {
            const seqStr = example.sequence.join(',');
            console.log(`${i+1}. ${seqStr} → ${example.predicted} ✓`);
        });
    } else {
        console.log('No correct predictions.');
    }

    console.log('\nSample Incorrect Predictions:');
    const incorrectSamples = results.filter(r => !r.correct);
    if (incorrectSamples.length > 0) {
        // Shuffle and take first 10
        const incorrectExamples = [...incorrectSamples]
            .sort(() => 0.5 - Math.random())
            .slice(0, 10);

        incorrectExamples.forEach((example, i) => {
            const seqStr = example.sequence.join(',');
            console.log(`${i+1}. ${seqStr} → ${example.predicted} (expected ${example.expected}) ✗`);
        });
    } else {
        console.log('No incorrect predictions.');
    }*/
}

// Run the main function
main().catch(error => {
    console.error(`Fatal error: ${error.message}`);
    process.exit(1);
});